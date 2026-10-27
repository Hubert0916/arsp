"""Verify that the checked-in inference files match the published manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "artifacts.sha256.json").read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: {name}")
print(f"Verified {len(manifest)} inference files.")
