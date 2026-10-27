"""Shared training/inference features and small-data probability adaptation."""

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
          "reasoning_effort", "harness_version", "subject_features_extra")


def normalize_text(value):
    return " ".join(str(value or "").split())


def subject_text(subject):
    return "; ".join(f"{key}: {subject.get(key, '')}" for key in FIELDS)


def text_numbers(text):
    text = normalize_text(text)
    prefix = text[:2000]
    n = max(1, len(prefix))
    return [np.log1p(len(text)) / 10, sum(c.isdigit() for c in prefix) / n,
            sum(ord(c) > 127 for c in prefix) / n,
            float(any(x in text[:6000] for x in ("```", "def ", "class ", "#include"))),
            float(any(x in text[:6000] for x in ("\\frac", "\\sum", " = "))),
            float("?" in prefix)]


def subject_numbers(subject):
    hashed = np.zeros(64, dtype=np.float32)
    for key in ("normalized_name", "provider", "harness", "reasoning_effort"):
        value = normalize_text(subject.get(key, "")).lower()
        if value:
            h = hashlib.sha256(f"{key}={value}".encode()).digest()
            hashed[int.from_bytes(h[:4], "big") % 64] += 1 if h[4] % 2 else -1
    dates = []
    for key in ("release_date", "access_date"):
        try:
            parsed = date.fromisoformat(str(subject.get(key, ""))[:10])
            dates += [(parsed - date(2020, 1, 1)).days / 3652.5, 1.0]
        except (ValueError, TypeError):
            dates += [0.0, 0.0]
    return np.concatenate([hashed, np.array(dates, dtype=np.float32)])


def combine_features(item_vectors, subject_vectors, item_numbers, subject_numeric):
    return np.concatenate([item_vectors, subject_vectors, item_vectors * subject_vectors,
                           item_numbers, subject_numeric], axis=1).astype(np.float32)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -40, 40)))


def fit_offset(logits, labels, strength):
    """MAP intercept: Bernoulli likelihood plus strength * offset² / 2."""
    if len(labels) == 0 or strength is None or np.isinf(strength):
        return 0.0
    logits, labels = np.asarray(logits), np.asarray(labels)
    offset = 0.0
    for _ in range(30):
        probabilities = sigmoid(logits + offset)
        gradient = float(np.sum(probabilities - labels) + strength * offset)
        curvature = float(np.sum(probabilities * (1 - probabilities)) + strength)
        step = float(np.clip(gradient / curvature, -2, 2))
        offset = float(np.clip(offset - step, -12, 12))
        if abs(step) < 1e-7:
            break
    return offset


class MiniLMEncoder:
    def __init__(self, path, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).to(self.device).eval()

    def encode(self, texts, batch_size=128, progress=False):
        import torch
        result = np.zeros((len(texts), 384), dtype=np.float32)
        order = np.argsort([min(len(text), 6000) for text in texts], kind="stable")
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                tokens = self.tokenizer([texts[i][:6000] for i in indices], padding=True,
                                        truncation=True, max_length=256, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                result[indices] = pooled.cpu().numpy()
                if progress and (start // batch_size % 100 == 0 or start + batch_size >= len(order)):
                    print(f"MiniLM encoded {min(start + batch_size, len(order)):,}/{len(order):,}", flush=True)
        return result


class NumpyHead:
    def __init__(self, artifact):
        artifact = Path(artifact)
        self.config = json.loads((artifact / "config.json").read_text())
        self.weights = dict(np.load(artifact / "head.npz"))

    def logits(self, features):
        x = np.clip((features - self.weights["mean"]) / self.weights["scale"], -10, 10)
        for i in range(self.config["layers"]):
            x = x @ self.weights[f"weight_{i}"].T + self.weights[f"bias_{i}"]
            if i < self.config["layers"] - 1:
                x = np.maximum(x, 0)
        return x.ravel().astype(np.float64)
