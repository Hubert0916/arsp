"""Construct policies that can inspect only supplied evidence and past inputs."""

from base_features import feature_key, subject_key
from adaptive_gain_policy import AdaptiveGainPolicy
from balanced_probe_policy import BalancedPolicy
from linear_gain_policy import LinearGainPolicy


def make_policy(gram, config, key=feature_key):
    kind = config["policy"]
    adaptive = {"adaptive_1": (1., 0), "adaptive_4": (4., 0),
                "adaptive_8": (8., 0), "adaptive_4_warm": (4., 16)}
    if kind in adaptive:
        power, warmup = adaptive[kind]
        return AdaptiveGainPolicy(gram, key, power=power, warmup=warmup)
    if kind == "balanced":
        return BalancedPolicy(gram, subject_key, key, strength=0.5)
    if kind not in ("stream_random", "diversity", "brier"):
        raise ValueError(f"Unknown streaming policy: {kind}")
    return LinearGainPolicy(gram, subject_key, key,
                            mix=0.0 if kind == "stream_random" else 1.0,
                            criterion=kind)
