"""Build the ASRP or Empirical Mean submission with files at the ZIP root."""
from pathlib import Path
import argparse
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["asrp", "empirical_mean"], default="asrp", nargs="?")
    args = parser.parse_args()
    source = ROOT / ("project/asrp" if args.method == "asrp" else "baselines/empirical_mean")
    if args.method == "asrp":
        for name in ["encoder/model.safetensors", "probe_encoder/onnx/model.onnx"]:
            if not (source / name).is_file():
                raise SystemExit(f"Missing required model: {name}")
    output = ROOT / "dist" / f"{args.method}.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, path.relative_to(source))
    print(f"Built {output} ({output.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
