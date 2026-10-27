"""Streaming kernel-mean matching against already seen candidate inputs."""

import hashlib
import json

import numpy as np


class BalancedPolicy:
    def __init__(self, gram, subject_key, feature_key, strength=0.5):
        self.gram, self.subject_key, self.feature_key = (
            gram,
            subject_key,
            feature_key,
        )
        self.strength = strength
        self.seen = {}

    def choose(self, input, evidence, context):
        subject, item = input
        benchmark = item.get("benchmark_id") or context["benchmark_id"]
        seen = self.seen.setdefault(benchmark, [])
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
        if len(past) < 2 or self.strength == 0:
            return bool(u < remaining / total)
        current = self.subject_key(subject)
        selected = [
            i
            for (s, i), _ in evidence
            if i.get("benchmark_id") == benchmark
            and self.subject_key(s) == current
        ]
        n = len(selected)
        k = self.gram([*selected, *past, item])
        # Candidate-dependent part of the negative squared MMD after adding
        # one point. The reference measure consists only of earlier inputs.
        score = 2 * k[n:, n:-1].mean(1)
        if n:
            score -= 2 * k[n:, :n].sum(1) / (n + 1)
        score -= np.diag(k[n:, n:]) / (n + 1)
        scale = max(float(score[:-1].std()), 1e-6)
        logweights = (
            np.clip((score - score[:-1].mean()) / scale, -2, 2) * self.strength
        )
        weights = np.exp(logweights)
        ratio = float(weights[-1] / weights[:-1].mean())
        return bool(u < min(1, remaining / total * np.clip(ratio, 0.25, 4)))
