"""Check a built submission on synthetic inputs; this is not the official evaluator."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def item(i):
    return {"item_content": f"What is {i} plus 2?", "item_features": "topic=arithmetic",
            "interactors": "", "benchmark_id": "synthetic-benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="asrp-smoke-") as temp:
        root = Path(temp)
        with ZipFile(args.archive) as archive:
            if "model.py" not in archive.namelist():
                raise ValueError("model.py must be at the ZIP root")
            if archive.testzip() is not None:
                raise ValueError("Corrupt ZIP")
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError("Unsafe ZIP path")
            archive.extractall(root)
        sys.path.insert(0, str(root))
        model = load_module(root / "model.py", "submission_model")
        subject = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date",
                                "harness", "harness_version", "reasoning_effort", "subject_features_extra"), "")
        subject["normalized_name"] = "synthetic-subject"
        evidence = [[[subject, item(i)], int(i % 3 != 0)] for i in range(31)]
        for budget in (0, 1, 3, 7, 15, 31):
            probability = float(model.predict([subject, item(100)], evidence[:budget]))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability at budget {budget}")
        if (root / "labeling.py").exists():
            hook = load_module(root / "labeling.py", "submission_labeling").acquisition_function
            selected = []
            for i in range(80):
                context = dict(subject_id="synthetic-subject", benchmark_id="synthetic-benchmark",
                               labels_acquired=len(selected), labels_remaining=31-len(selected),
                               max_labels=31, items_remaining=80-i)
                decision = hook([subject, item(i)], labeled=selected, context=context)
                if not isinstance(decision, bool):
                    raise ValueError("Acquisition must return bool")
                if decision:
                    selected.append([[subject, item(i)], int(i % 3 != 0)])
            if len(selected) != 31:
                raise ValueError(f"Acquired {len(selected)} labels instead of 31")
            print("Streaming acquisition passed (31 labels from 80 synthetic candidates).")
        print("ZIP and six-budget prediction smoke test passed. No performance score is estimated.")


if __name__ == "__main__":
    main()
