"""Frozen text embedding used by topic-cluster calibration."""
import numpy as np


def normalize_text(value):
    return " ".join(str(value or "").split())


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


