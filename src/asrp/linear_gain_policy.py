"""Portable streaming covariance policy, independent of research files."""

import hashlib
import json

import numpy as np


class LinearGainPolicy:
    def __init__(self, gram, subject_key, feature_key, mix, criterion="brier"):
        self.gram, self.subject_key, self.feature_key = (
            gram,
            subject_key,
            feature_key,
        )
        self.mix, self.criterion = mix, criterion
        self.seen, self.trace = {}, []

    def choose(self, input, evidence, context):
        subject, item = input
        bench = item.get("benchmark_id") or context["benchmark_id"]
        seen = self.seen.setdefault(bench, [])
        past = list({self.feature_key(i): i for i in seen[-32:]}.values())
        seen.append(item.copy())
        remaining, total = (
            context["labels_remaining"],
            context["items_remaining"],
        )
        if remaining <= 0:
            return False
        if total <= remaining:
            return True
        raw = json.dumps(
            [context, input], sort_keys=True, separators=(",", ":")
        )
        u = (
            int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")
            / 2**64
        )
        if not past or self.mix == 0:
            return bool(u < remaining / total)
        records = [
            (s, i, y)
            for (s, i), y in evidence
            if i.get("benchmark_id") == bench
        ]
        current = self.subject_key(subject)
        same = np.array(
            [self.subject_key(s) == current for s, i, y in records]
        )
        who = [self.subject_key(s) for s, i, y in records]
        same_train = np.array(
            [[a == b for a in who] for b in who], float
        ).reshape(len(who), len(who))
        query = [*past, item]
        gram = self.gram([*[i for s, i, y in records], *query])
        n = len(records)
        k = (gram[:n, :n] + same_train) / 4
        cross = (gram[n:, :n] + same[None, :]) / 4
        covariance = (gram[n:, n:] + 1) / 4
        if n:
            covariance -= cross @ np.linalg.solve(k + np.eye(n), cross.T)
        variance = np.maximum(np.diag(covariance), 0)
        gains = (
            (covariance[:-1] ** 2).mean(0) / (1 + variance)
            if self.criterion == "brier"
            else variance
        )
        ratio = float(gains[-1] / max(gains[:-1].mean(), 1e-12))
        weight = 1 - self.mix + self.mix * np.clip(ratio, 0.25, 4)
        accepted = bool(u < min(1, remaining / total * weight))
        self.trace.append(
            dict(
                gain=float(gains[-1]),
                reference=float(gains[:-1].mean()),
                labels=n,
                accepted=accepted,
            )
        )
        return accepted
