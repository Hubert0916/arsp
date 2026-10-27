"""Budget-wise ensemble of kernel Probe and amortized Rasch.

The kernel Probe is strongest after enough labels have been acquired, while
the amortized difficulty model is more stable at budget zero and one.  The
fixed blend was selected on benchmark-held-out validation only.
"""

from functools import lru_cache
import json
import re
from pathlib import Path

import numpy as np

from ml_features import MiniLMEncoder, normalize_text, text_numbers
from kernel_probe import get_predictor
from base_features import feature_key, subject_key
from evidence_features import lexical_features
from window_features import WindowEncoder
from project_core import fit_evidence, predict_fitted, initial
from collections import OrderedDict

HERE = Path(__file__).resolve().parent
BUDGETS = (0, 1, 3, 7, 15, 31)
PROBE_WEIGHTS = (0.00, 0.00, 0.20, 0.40, 1.00, 1.00)


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -40., 40.)))


def subject_feature_vector(subject, vocabulary):
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    for field in ("normalized_name", "provider", "harness", "harness_version", "reasoning_effort"):
        key = field + "=" + str(subject.get(field, ""))
        if key in vocabulary:
            vector[vocabulary[key]] = 1.
    return vector



# The competition exposes item_features at runtime.  These helpers use only
# public-training aggregates and the current query/evidence metadata.
def _feature_tokens(raw):
    if not raw:
        return []
    out = []
    for part in str(raw).split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key, value = key.strip().lower(), re.sub(r"\s+", " ", value.strip().lower())
        if not value or len(value) > 160:
            continue
        out.append(f"{key}={value}")
        for word in re.findall(r"[a-z][a-z0-9_-]{2,}", value):
            if word not in {"true", "false", "none", "null", "english", "test", "dev", "train"}:
                out.append(f"{key}~{word}")
    return list(dict.fromkeys(out))


def _explicit_topics(item):
    """Return explicit topic keys in the source metadata order."""
    preferred = {"category", "task", "domain", "topic", "subject", "area",
                 "track", "dataset", "difficulty", "format"}
    values = []
    for part in str(item.get("item_features", "") or "").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = re.sub(r"\s+", " ", value.strip().lower())
        if key in preferred and value:
            values.append(f"{key}={value[:100]}")
    return values


def _explicit_topic(item):
    topics = _explicit_topics(item)
    return topics[0] if topics else ""


def _topic_cluster(vector, calibration):
    """Assign a missing-metadata item to a frozen MiniLM topic centroid."""
    centers = calibration.get("centroids") if calibration else None
    if not centers:
        return ""
    x = np.asarray(vector[:384], dtype=np.float64)
    x /= max(float(np.linalg.norm(x)), 1e-8)
    c = np.asarray(centers, dtype=np.float64)
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
    return str(int(np.argmax(c @ x)))


def _budget_index(n):
    if n <= BUDGETS[0]:
        return 0
    if n >= BUDGETS[-1]:
        return len(BUDGETS) - 1
    return min(range(len(BUDGETS)), key=lambda i: abs(BUDGETS[i] - n))


def _calibration_adjust(probability, item, vector, n, calibration):
    """Apply a strongly shrunk topic residual learned from public data."""
    if not calibration:
        return float(probability)
    topics = _explicit_topics(item)
    table = calibration.get("topic", {})
    known = [topic for topic in topics if topic in table]
    key = known[0] if known else ""
    if not key:
        key = "cluster=" + _topic_cluster(vector, calibration)
        table = calibration.get("cluster", {})
    stats = table.get(key.split("=", 1)[1] if key.startswith("cluster=") else key)
    if stats is None:
        return float(probability)
    prior_n = float(calibration.get("cluster_prior_n", 200.0)
                    if key.startswith("cluster=")
                    else calibration.get("topic_prior_n", 40.0))
    idx = _budget_index(n)
    residual = float(stats.get("residual", [0.] * len(BUDGETS))[idx])
    shrink = float(stats.get("n", 0.0)) / (float(stats.get("n", 0.0)) + prior_n)
    limit = float(calibration.get("cluster_max_adjust", .03)
                  if key.startswith("cluster=")
                  else calibration.get("topic_max_adjust", .06))
    return float(np.clip(probability + np.clip(shrink * residual, -limit, limit), 0., 1.))


def _feature_prior(subject, item, artifact):
    """Predict a query probability from subject and visible metadata."""
    global_s, global_n = artifact["global"]
    global_p = float(global_s / max(global_n, 1.0))
    name = str(subject.get("normalized_name", ""))
    sg, sn = artifact["subject_global"].get(name, artifact["global"])
    subject_p = (float(sg) + 8.0 * global_p) / (float(sn) + 8.0)
    cells = artifact["subject_token"].get(name, {})
    values = []
    for token in _feature_tokens(item.get("item_features", "")):
        token_stats = artifact["token_global"].get(token)
        if token_stats is None:
            continue
        ts, tn = token_stats
        token_p = float(ts / max(tn, 1.0))
        cell = cells.get(token)
        if cell is None:
            reliability = min(float(tn), 300.0) / (min(float(tn), 300.0) + 100.0)
            values.append((token_p, reliability))
        else:
            cs, cn = cell
            cell_p = (float(cs) + 8.0 * token_p) / (float(cn) + 8.0)
            reliability = min(float(cn), 80.0) / (min(float(cn), 80.0) + 12.0)
            values.append((cell_p, reliability))
    if not values:
        return float(np.clip(subject_p, .03, .97))
    weights = np.asarray([x[1] for x in values], dtype=np.float64)
    probs = np.asarray([x[0] for x in values], dtype=np.float64)
    return float(np.clip((2.0 * subject_p + float(weights @ probs)) /
                         (2.0 + float(weights.sum())), .03, .97))


def _feature_adapt(query, support_items, labels, base, concentration=2.0):
    """Transfer acquired outcomes only across exact visible metadata matches."""
    qtokens = set(_feature_tokens(query.get("item_features", "")))
    if not qtokens or not support_items:
        return float(base)
    weights = []
    for support in support_items:
        overlap = qtokens.intersection(_feature_tokens(support.get("item_features", "")))
        score = 0.0
        for token in overlap:
            key = token.split("=", 1)[0].split("~", 1)[0]
            score += 1.0 if key in {"category", "task", "domain", "track", "subject", "topic", "area", "field"} else .35
        weights.append(min(score, 2.0))
    weights = np.asarray(weights, dtype=np.float64)
    if float(weights.sum()) <= 1e-9:
        return float(base)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.clip((concentration * base + float(weights @ labels)) /
                         (concentration + float(weights.sum())), .02, .98))


def update_ability(prior, difficulty, labels, variance):
    if not labels:
        return float(prior)
    difficulty, labels = np.asarray(difficulty, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    theta, precision = float(prior), 1. / max(float(variance), 1e-6)
    for _ in range(30):
        p = sigmoid(theta - difficulty)
        gradient = float(np.sum(labels - p) - precision * (theta - prior))
        curvature = float(np.sum(p * (1. - p)) + precision)
        step = np.clip(gradient / max(curvature, 1e-6), -2., 2.)
        theta += step
        if abs(step) < 1e-7:
            break
    return float(np.clip(theta, -12., 12.))


@lru_cache(maxsize=1)
def _resources():
    encoder = MiniLMEncoder(HERE / "encoder")
    difficulty = dict(np.load(HERE / "difficulty_model.npz"))
    prior = json.loads((HERE / "subject_prior.json").read_text())
    # binary_subject_prior is keyed by the runtime normalized_name field and is
    # therefore usable for hidden sources where the full subject dict varies.
    prior["binary"] = json.loads((HERE / "binary_subject_prior.json").read_text())
    feature_path = HERE / "feature_prior.json"
    prior["feature_prior"] = json.loads(feature_path.read_text()) if feature_path.exists() else None
    calibration_path = HERE / "topic_calibration.json"
    prior["topic_calibration"] = (json.loads(calibration_path.read_text())
                                   if calibration_path.exists() else None)
    vocabulary = json.loads((HERE / "subject_vocab.json").read_text())
    config = json.loads((HERE / "runtime_config.json").read_text())
    return encoder, difficulty, prior, vocabulary, config


def _item_vector(encoder, item):
    text = normalize_text(item.get("item_content", ""))
    return np.concatenate([encoder.encode([text])[0], np.asarray(text_numbers(text), dtype=np.float32)])


def _difficulty(vector, artifact):
    x = np.clip((vector - artifact["mean"]) / artifact["scale"], -10, 10)
    layers = sum(k.startswith("weight_") for k in artifact)
    for i in range(layers):
        x = x @ artifact[f"weight_{i}"].T + artifact[f"bias_{i}"]
        if i + 1 < layers:
            x = np.maximum(x, 0)
    return float(x.ravel()[0])


def predict(input: list, labeled: list | None = None) -> float:
    """Return the validation-selected budget-wise Probe ensemble."""
    evidence = labeled or []
    hybrid, query_vector = _predict_hybrid(input, evidence, return_vector=True)
    probe = float(get_predictor().predict(input, evidence))
    subject, item = input
    benchmark = item.get("benchmark_id")
    n = sum(1 for (observed_subject, observed_item), label in evidence
            if observed_subject == subject and observed_item.get("benchmark_id") == benchmark
            and label in (0, 1))
    # The platform calls the predictor at the six official budgets.  Interpolate
    # between adjacent entries defensively for local smoke tests or future API
    # variants that reveal a different count.
    if n <= BUDGETS[0]:
        weight = PROBE_WEIGHTS[0]
    elif n >= BUDGETS[-1]:
        weight = PROBE_WEIGHTS[-1]
    else:
        right = next(i for i, budget in enumerate(BUDGETS) if budget >= n)
        if BUDGETS[right] == n:
            weight = PROBE_WEIGHTS[right]
        else:
            left = right - 1
            fraction = (n - BUDGETS[left]) / (BUDGETS[right] - BUDGETS[left])
            weight = PROBE_WEIGHTS[left] + fraction * (PROBE_WEIGHTS[right] - PROBE_WEIGHTS[left])
    calibration = _resources()[2].get("topic_calibration")
    # A source-level late-budget guard reduces the kernel contribution only
    # where public validation showed a repeatable regression against smoothing.
    guard = 0.0
    if calibration is not None and n >= 15:
        stats = calibration.get("source", {}).get(str(benchmark), {})
        guard = float(np.clip(stats.get("guard", 0.0), 0.0, 0.35))
    effective_weight = weight * (1.0 - guard)
    probability = effective_weight * probe + (1.0 - effective_weight) * hybrid
    probability = _calibration_adjust(probability, item, query_vector, n, calibration)
    return float(np.clip(probability, 0.0, 1.0))


def _predict_hybrid(input: list, labeled: list, return_vector: bool = False):
    subject, item = input
    encoder, difficulty, prior, vocabulary, config = _resources()
    binary = prior.get("binary", {"offsets": {}})
    offset = float(binary.get("offsets", {}).get(subject.get("normalized_name", ""), 0.0))
    base_p = float(np.clip(.5 + .5 * offset, .05, .95))
    ability_prior = float(np.log(base_p / (1.0 - base_p))) + config["prior_shift"]
    feature_artifact = prior.get("feature_prior")
    query_prior = (_feature_prior(subject, item, feature_artifact)
                   if feature_artifact is not None else base_p)
    query_vector = _item_vector(encoder, item)
    query_difficulty = _difficulty(query_vector, difficulty)
    support_difficulties, labels, support_items = [], [], []
    benchmark = item.get("benchmark_id")
    for (observed_subject, observed_item), label in labeled or []:
        if observed_subject != subject or observed_item.get("benchmark_id") != benchmark:
            continue
        if label not in (0, 1):
            raise ValueError("Acquired outcomes must be binary.")
        support_difficulties.append(_difficulty(_item_vector(encoder, observed_item), difficulty))
        support_items.append(observed_item)
        labels.append(int(label))
    clip = config["difficulty_clip"]
    if clip is not None:
        query_difficulty = float(np.clip(query_difficulty, -clip, clip))
        support_difficulties = [float(np.clip(x, -clip, clip)) for x in support_difficulties]
    ability = update_ability(ability_prior, support_difficulties[:31], labels[:31], config["variance"])
    hierarchical = float(sigmoid(ability - query_difficulty))
    p_source = float((1. - config["source_prior_weight"]) * .5 +
                     config["source_prior_weight"] * query_prior)
    n = len(labels[:31])
    if n:
        global_smoothed = (sum(labels[:31]) + config["kappa"] * p_source) / (n + config["kappa"])
        match_weight = float(config.get("metadata_match_weight", 0.0))
        if match_weight > 0.0:
            matched = _feature_adapt(
                item, support_items, labels[:31], query_prior,
                concentration=float(config.get("metadata_match_concentration", 2.0)))
            smoothed = match_weight * matched + (1.0 - match_weight) * global_smoothed
        else:
            smoothed = global_smoothed
    else:
        smoothed = p_source
    if "lambdas" in config:
        try:
            lam = float(config["lambdas"][BUDGETS.index(n)])
        except (ValueError, IndexError, TypeError):
            lam = config["lambda_max"] * n / (n + config["lambda_tau"])
    else:
        lam = config["lambda_max"] * n / (n + config["lambda_tau"])
    result = float((1. - lam) * smoothed + lam * hierarchical)
    return (result, query_vector) if return_vector else result


class Predictor:
    """Compatibility adapter for the standalone kernel Probe tests.

    The shipped ``predict`` function above uses the ensemble.  Keeping this
    small class preserves the original support-only Probe API for local
    numerical checks and downstream users that instantiate a predictor.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        prior_path = self.directory / "binary_subject_prior.json"
        self.prior = json.loads(prior_path.read_text()) if prior_path.exists() else {"offsets": {}}
        self.encoder, self.vectors, self.fits = None, {}, OrderedDict()

    def offset(self, subject):
        return self.prior.get("offsets", {}).get(subject.get("normalized_name", ""), 0.0)

    def initial(self, subject):
        return float(initial(self.offset(subject), self.config))

    def features(self, items):
        missing = {feature_key(i): i for i in items if feature_key(i) not in self.vectors}
        if missing:
            if self.encoder is None:
                self.encoder = WindowEncoder(self.directory / "encoder", threads=2)
            values = list(missing.values())
            encoded = self.encoder.encode(values)[:, :384]
            for item, vector in zip(values, encoded):
                self.vectors[feature_key(item)] = np.r_[vector, lexical_features(item, bins=8192, mode="tokens")]
        return np.stack([self.vectors[feature_key(i)] for i in items])

    def predict(self, input, labeled=None):
        subject, item = input
        if not labeled:
            return self.initial(subject)
        benchmark = item.get("benchmark_id")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError("An anonymous benchmark_id is required")
        records, subjects = [], {}
        for (observed_subject, observed_item), label in labeled:
            if observed_item.get("benchmark_id") != benchmark:
                continue
            if label not in (0, 1):
                raise ValueError("Revealed labels must be binary")
            key = subject_key(observed_subject)
            subjects[key] = observed_subject
            records.append((key, feature_key(observed_item), int(label), observed_item))
        current = subject_key(subject)
        if current not in subjects:
            return self.initial(subject)
        records.sort(key=lambda r: r[:3])
        fingerprint = (benchmark, tuple(r[:3] for r in records))
        if fingerprint not in self.fits:
            names = sorted(subjects)
            positions = {name: j for j, name in enumerate(names)}
            who = np.array([positions[r[0]] for r in records])
            labels = np.array([r[2] for r in records], float)
            values = self.features([r[3] for r in records])
            offsets = np.array([self.offset(subjects[k]) for k in names])
            self.fits[fingerprint] = positions, fit_evidence(
                values[:, :384], values[:, 384:], labels, who, offsets, self.config
            )
            if len(self.fits) > 8:
                self.fits.popitem(last=False)
        self.fits.move_to_end(fingerprint)
        positions, state = self.fits[fingerprint]
        query = self.features([item])
        return float(predict_fitted(state, query[:, :384], query[:, 384:],
                                    np.array([positions[current]]), self.config)[0])
