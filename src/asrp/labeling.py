"""Uniform random streaming acquisition with a reproducible hash draw."""
import hashlib
import json


def acquisition_function(input: list[dict], labeled: list | None = None,
                         context: dict | None = None) -> bool:
    """Acquire with probability labels_remaining / items_remaining."""
    if context is None:
        raise ValueError("Streaming acquisition requires evaluator context")
    remaining = context["labels_remaining"]
    total = context["items_remaining"]
    if remaining <= 0 or total <= 0:
        return False
    if total <= remaining:
        return True
    raw = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") / 2**64
    return bool(uniform < remaining / total)
