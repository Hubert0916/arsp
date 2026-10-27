"""Frozen MiniLM features from up to four evenly spaced text windows."""

from __future__ import annotations

import numpy as np
from base_features import TextEncoder, item_text, numeric_features


def window_starts(length, width=254, maximum=4):
    count = min(maximum, max(1, int(np.ceil(length / width))))
    return np.unique(np.linspace(0, max(0, length - width), count).astype(int))


class WindowEncoder(TextEncoder):
    """Encode each window, normalize, average windows, then normalize again."""

    def encode(self, items):
        sequences, owners = [], []
        for owner, encoded in enumerate(
            self.tokenizer.encode_batch(
                [item_text(item) for item in items], add_special_tokens=False
            )
        ):
            for start in window_starts(len(encoded.ids)):
                sequences.append(
                    [
                        self.cls,
                        *encoded.ids[start : start + 254],
                        self.sep,
                    ]
                )
                owners.append(owner)
        pieces = []
        for start in range(0, len(sequences), 32):
            batch = sequences[start : start + 32]
            width = max(map(len, batch))
            ids = np.full((len(batch), width), self.pad, dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, sequence in enumerate(batch):
                ids[row, : len(sequence)] = sequence
                mask[row, : len(sequence)] = 1
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            }
            names = {entry.name for entry in self.session.get_inputs()}
            hidden = self.session.run(
                None, {k: v for k, v in inputs.items() if k in names}
            )[0]
            pooled = (hidden * mask[:, :, None]).sum(axis=1)
            pooled /= mask.sum(axis=1, keepdims=True)
            pooled /= np.maximum(
                np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
            )
            pieces.append(pooled)
        values = np.vstack(pieces)
        pooled = np.zeros((len(items), values.shape[1]))
        np.add.at(pooled, owners, values)
        pooled /= np.maximum(
            np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
        )
        return np.column_stack(
            [
                pooled,
                np.stack([numeric_features(item) for item in items]),
            ]
        )
