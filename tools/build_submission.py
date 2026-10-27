"""Combine source and frozen artifacts into a flat competition submission ZIP."""
from pathlib import Path
import argparse
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["asrp", "empirical_mean"], default="asrp", nargs="?")
    args = parser.parse_args()
    sources = ([ROOT / "src/asrp", ROOT / "artifacts/asrp"] if args.method == "asrp"
               else [ROOT / "baselines/empirical_mean"])
    if args.method == "asrp":
        for name in ["encoder/model.safetensors", "probe_encoder/onnx/model.onnx"]:
            if not (sources[1] / name).is_file():
                raise SystemExit(f"Missing required model: {name}")
    files = {}
    for source in sources:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            name = path.relative_to(source).as_posix()
            if name in files:
                raise SystemExit(f"Duplicate ZIP path: {name}")
            files[name] = path
    output = ROOT / "dist" / f"{args.method}.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
    print(f"Built {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
