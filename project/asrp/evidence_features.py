"""Deterministic full-text lexical and structural features, without labels."""

import re
import zlib
from collections import Counter

import numpy as np
from base_features import item_text


def lexical_features(item, bins=512, mode="tokens"):
    text = item_text(item).lower()
    pattern = r"[\w]+" if mode == "words" else r"[\w]+|[^\w\s]"
    words = re.findall(pattern, text)
    counts = Counter(words)
    counts.update(a + " " + b for a, b in zip(words, words[1:]))
    vector = np.zeros(bins)
    for word, count in counts.items():
        hashed = zlib.crc32(word.encode("utf-8"))
        sign = 1 if hashed & (1 << 31) else -1
        vector[hashed % bins] += sign * (1 + np.log(count))
    return vector / max(np.linalg.norm(vector), 1e-12)


def structure_features(item):
    text = item.get("item_content", "")
    length = max(1, len(text))
    lines = text.splitlines() or [""]
    return np.array(
        [
            np.log1p(len(text)) / 10,
            np.log1p(len(lines)) / 6,
            np.log1p(len(text.split())) / 9,
            np.log1p(max(map(len, lines))) / 8,
            sum(c.isdigit() for c in text) / length,
            sum(not c.isascii() for c in text) / length,
            sum(c in "{}[]();=\\$" for c in text) / length,
            text.count("\n") / length,
        ],
        dtype=float,
    )


def combine_features(semantic, lexical, structure, config):
    pieces = [
        config.get("semantic_scale", 1) * np.asarray(semantic, dtype=float)
    ]
    if config.get("lexical_scale", 0):
        pieces.append(config["lexical_scale"] * np.asarray(lexical))
    if config.get("structure_scale", 0):
        pieces.append(config["structure_scale"] * np.asarray(structure))
    return np.concatenate(pieces, axis=-1)
