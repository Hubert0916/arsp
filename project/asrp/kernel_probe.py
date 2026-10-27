"""Kernel Probe predictor used by the budget-wise ensemble.

This is the support-conditioned part of the submission.  It is deliberately
kept separate from the amortized difficulty model in ``model.py`` so the
ensemble can trust the two signals at different label budgets.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from base_features import feature_key, subject_key
from evidence_features import lexical_features
from threadpoolctl import threadpool_limits
from window_features import WindowEncoder

from project_core import fit_evidence, predict_fitted, initial


class Predictor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "probe_config.json").read_text())
        self.prior = (json.loads((self.directory / "binary_subject_prior.json").read_text())
                      if self.config["carry"] else {"offsets": {}})
        self.encoder, self.vectors, self.fits = None, {}, OrderedDict()

    def offset(self, subject):
        return self.prior["offsets"].get(subject.get("normalized_name", ""), 0.0)

    def initial(self, subject):
        return float(initial(self.offset(subject), self.config))

    def features(self, items):
        missing = {feature_key(i): i for i in items if feature_key(i) not in self.vectors}
        if missing:
            if self.encoder is None:
                self.encoder = WindowEncoder(self.directory / "probe_encoder", threads=2)
            values = list(missing.values())
            encoded = self.encoder.encode(values)[:, :384]
            for item, vector in zip(values, encoded):
                self.vectors[feature_key(item)] = np.r_[
                    vector, lexical_features(item, bins=8192, mode="tokens")
                ]
        return np.stack([self.vectors[feature_key(i)] for i in items])

    def predict(self, input, labeled=None):
        subject, item = input
        if not labeled:
            return self.initial(subject)
        benchmark = item.get("benchmark_id")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError("An anonymous benchmark_id is required")
        records, subjects = [], {}
        for (s, i), y in labeled:
            if i.get("benchmark_id") != benchmark:
                continue
            if y not in (0, 1):
                raise ValueError("Revealed labels must be binary")
            key = subject_key(s)
            subjects[key] = s
            records.append((key, feature_key(i), int(y), i))
        current = subject_key(subject)
        if current not in subjects:
            return self.initial(subject)
        records.sort(key=lambda r: r[:3])
        fingerprint = (benchmark, tuple(r[:3] for r in records))
        with threadpool_limits(limits=2):
            if fingerprint not in self.fits:
                names = sorted(subjects)
                positions = {name: j for j, name in enumerate(names)}
                who = np.array([positions[r[0]] for r in records])
                y = np.array([r[2] for r in records], float)
                x = self.features([r[3] for r in records])
                offsets = np.array([self.offset(subjects[k]) for k in names])
                state = fit_evidence(x[:, :384], x[:, 384:], y, who, offsets, self.config)
                self.fits[fingerprint] = positions, state
                if len(self.fits) > 8:
                    self.fits.popitem(last=False)
            self.fits.move_to_end(fingerprint)
            positions, state = self.fits[fingerprint]
            q = self.features([item])
            return float(predict_fitted(state, q[:, :384], q[:, 384:],
                                       np.array([positions[current]]), self.config)[0])


_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor(Path(__file__).resolve().parent)
    return _predictor


def predict(input, labeled=None):
    return get_predictor().predict(input, labeled)
