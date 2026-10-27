"""Reproduce the published scores from pinned public data and frozen weights."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Require all pinned data in the Hugging Face cache")
    parser.add_argument("--prepared", action="store_true", help="Reuse data/proxy.jsonl (its checksum is still verified)")
    args = parser.parse_args()
    def run(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / "tools" / script), *arguments], cwd=ROOT, check=True)
    run("verify_artifacts.py")
    run("test_proxy.py")
    if not args.prepared:
        run("prepare_proxy.py", *(["--cache-only"] if args.cache_only else []))
    run("build_submission.py", "asrp")
    run("evaluate_proxy.py")
    run("score_proxy.py", "results/reproduced", "--compare", "results/proxy")
    print("Reproduction complete: results/reproduced/summary.json")


if __name__ == "__main__":
    main()
