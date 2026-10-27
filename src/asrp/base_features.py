"""Frozen text features shared by offline training and the submission.

Only runtime-visible text is encoded. No benchmark alias, answer, outcome,
or original ID enters the encoder. Learned preprocessing is fit separately
inside each training fold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_FILES = ("tokenizer.json", "onnx/model.onnx")
MAX_TOKENS = 256
FEATURE_VERSION = 1


def item_text(item: dict) -> str:
    return (
        "Task:\n"
        + item.get("item_content", "")
        + "\nTask metadata:\n"
        + item.get("item_features", "")
        + "\nInteractors:\n"
        + item.get("interactors", "")
    )


def feature_key(item: dict) -> str:
    return hashlib.sha256(item_text(item).encode("utf-8")).hexdigest()


def subject_key(subject: dict) -> str:
    text = json.dumps(subject, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def numeric_features(item: dict) -> np.ndarray:
    text = item.get("item_content", "")
    return np.array(
        [
            np.log1p(len(text)),
            np.log1p(text.count("\n")),
            sum(c.isdigit() for c in text) / max(len(text), 1),
        ],
        dtype=np.float64,
    )


class TextEncoder:
    """Initialize the frozen ONNX model used by WindowEncoder."""

    def __init__(self, directory: Path, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.cls = self.tokenizer.token_to_id("[CLS]")
        self.sep = self.tokenizer.token_to_id("[SEP]")
        self.pad = self.tokenizer.token_to_id("[PAD]")
        if any(x is None for x in (self.cls, self.sep, self.pad)):
            raise ValueError("Expected a BERT-compatible tokenizer")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "onnx/model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
