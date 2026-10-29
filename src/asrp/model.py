"""ASRP: metadata-smoothed public priors with support-conditioned kernel adaptation."""
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np

from text_encoder import MiniLMEncoder, normalize_text
from kernel_probe import get_predictor

HERE = Path(__file__).resolve().parent
BUDGETS = (0, 1, 3, 7, 15, 31)


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


@lru_cache(maxsize=1)
def _resources():
    def read(name, fallback=None):
        path = HERE / name
        return json.loads(path.read_text()) if path.exists() else fallback
    return {
        "encoder": MiniLMEncoder(HERE / "encoder"),
        "binary": read("binary_subject_prior.json", {"offsets": {}}),
        "feature_prior": read("feature_prior.json"),
        "topic_calibration": read("topic_calibration.json"),
        "config": read("runtime_config.json"),
    }


def _item_embedding(encoder, item):
    return encoder.encode([normalize_text(item.get("item_content", ""))])[0]


def _predict_smooth(input, evidence):
    """Update the query's frozen metadata prior with matching revealed outcomes."""
    subject, item = input
    resources = _resources()
    feature = resources["feature_prior"]
    if feature is None:
        offset = resources["binary"].get("offsets", {}).get(subject.get("normalized_name", ""), 0.)
        prior = float(np.clip(.5 + .5 * offset, .05, .95))
    else:
        prior = _feature_prior(subject, item, feature)
    labels = []
    for (observed_subject, observed_item), label in evidence:
        if observed_subject != subject or observed_item.get("benchmark_id") != item.get("benchmark_id"):
            continue
        if label not in (0, 1):
            raise ValueError("Acquired outcomes must be binary.")
        labels.append(int(label))
    labels = labels[:31]
    n = len(labels)
    kappa = resources["config"]["kappa"]
    return (float((sum(labels) + kappa * prior) / (n + kappa)) if n else prior), n


def _kernel_weight(n):
    weights = _resources()["config"]["probe_weights"]
    return float(np.interp(n, BUDGETS, weights))


def _prediction_components(input, labeled=None):
    evidence = labeled or []
    smooth, n = _predict_smooth(input, evidence)
    kernel = float(get_predictor().predict(input, evidence))
    vector = _item_embedding(_resources()["encoder"], input[1])
    return smooth, kernel, vector, n


def _combine(input, smooth, kernel, vector, n):
    weight = _kernel_weight(n)
    probability = weight * kernel + (1. - weight) * smooth
    probability = _calibration_adjust(probability, input[1], vector, n,
                                      _resources()["topic_calibration"])
    return float(np.clip(probability, 0., 1.))


def predict(input: list, labeled: list | None = None) -> float:
    """Return a probability using frozen priors and only the revealed evidence."""
    return _combine(input, *_prediction_components(input, labeled))
