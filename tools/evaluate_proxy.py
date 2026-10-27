"""Run the packaged predictors with identical sequential acquired evidence."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from zipfile import ZipFile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from prepare_proxy import ROOT, sha, write_json
from smoke_test import load_module


def finite_probability(p):
    p = float(p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f"Invalid prediction: {p}")
    return p


def evaluate(groups, model, labeling, baseline, output, budgets, cache_vectors=True):
    # Cache a pure, frozen text-to-vector function. No label or support fit is cached here.
    # This avoids repeatedly encoding the same support text for every query.
    original_vector = model._item_vector
    if cache_vectors:
        @lru_cache(maxsize=4096)
        def vector(encoder, content):
            value = original_vector(encoder, {"item_content": content})
            value.flags.writeable = False
            return value
        model._item_vector = lambda encoder, item: vector(encoder, item.get("item_content", ""))
    start = time.monotonic()
    with (output / "predictions.jsonl").open("w") as predictions, (output / "acquisition.jsonl").open("w") as acquisition:
        for number, group in enumerate(groups, 1):
            # Groups have distinct benchmark aliases; reset transient state to bound memory.
            labeling._policy = None
            model.get_predictor().vectors.clear()
            model.get_predictor().fits.clear()
            support, position, decisions = [], 0, []
            records = [{"source": group["source"], "subject_id": group["subject_id"],
                        "item_id": row["item_id"], "response_id": row["response_id"], "y": row["label"],
                        "asrp": [], "empirical_mean": []} for row in group["evaluation"]]
            acquired_ids = []
            for budget in budgets:
                while len(support) < budget:
                    if position >= len(group["acquisition"]):
                        raise ValueError("Acquisition stream exhausted before budget")
                    candidate = group["acquisition"][position]
                    context = {"subject_id": group["subject_alias"], "benchmark_id": group["benchmark_alias"],
                               "labels_acquired": len(support), "labels_remaining": 31-len(support),
                               "max_labels": 31, "items_remaining": len(group["acquisition"])-position}
                    # Only the current candidate and labels already purchased reach the hook.
                    chosen = labeling.acquisition_function(candidate["input"], labeled=support.copy(), context=context.copy())
                    if not isinstance(chosen, bool):
                        raise ValueError("Acquisition hook must return bool")
                    decisions.append({"position": position, "item_id": candidate["item_id"], "acquired": chosen})
                    if chosen:
                        support.append([candidate["input"], candidate["label"]])
                        acquired_ids.append(candidate["item_id"])
                    position += 1
                for row, record in zip(group["evaluation"], records):
                    record["asrp"].append(finite_probability(model.predict(row["input"], support.copy())))
                    record["empirical_mean"].append(finite_probability(baseline.predict(row["input"], support.copy())))
                # Verify memoization against the unmodified packaged code on the first group.
                if number == 1 and cache_vectors:
                    cached_vector = model._item_vector
                    model._item_vector = original_vector
                    direct = finite_probability(model.predict(group["evaluation"][0]["input"], support.copy()))
                    model._item_vector = cached_vector
                    if abs(direct-records[0]["asrp"][-1]) > 1e-12:
                        raise ValueError("Memoization changed packaged predictions")
            for record in records:
                predictions.write(json.dumps(record, separators=(",", ":")) + "\n")
            acquisition.write(json.dumps({"source": group["source"], "subject_id": group["subject_id"],
                "acquired_item_ids": acquired_ids, "labels": [r[1] for r in support],
                "decisions": decisions}, separators=(",", ":")) + "\n")
            predictions.flush()
            acquisition.flush()
            print(f"Evaluated {number}/{len(groups)}: {group['source']} ({time.monotonic()-start:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/proxy.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    parser.add_argument("--max-groups", type=int, help="Development subset only; not a report result")
    parser.add_argument("--no-vector-cache", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments/proxy_manifest.json").read_text())
    if sha(args.data) != manifest["prepared_sha256"]:
        raise ValueError("Prepared data differs from the verified split")
    groups = [json.loads(line) for line in args.data.read_text().splitlines()]
    if args.max_groups is not None:
        groups = groups[:args.max_groups]
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asrp-evaluate-") as directory:
        root = Path(directory)
        with ZipFile(ROOT / "dist/asrp.zip") as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError("Unsafe ZIP path")
            archive.extractall(root)
        sys.path.insert(0, str(root))
        model = load_module(root / "model.py", "asrp_submission")
        labeling = load_module(root / "labeling.py", "asrp_labeling")
        baseline = load_module(ROOT / "baselines/empirical_mean/model.py", "empirical_baseline")
        evaluate(groups, model, labeling, baseline, args.output, manifest["config"]["budgets"], not args.no_vector_cache)
    provenance = {"config": manifest["config"], "n_groups": len(groups), "data_sha256": sha(args.data),
                  "split_manifest_sha256": sha(ROOT / "experiments/proxy_manifest.json"),
                  "inference_manifest_sha256": sha(ROOT / "artifacts/manifest.sha256.json"),
                  "lockfile_sha256": sha(ROOT / "uv.lock"), "python": platform.python_version(),
                  "files": {p.name: sha(p) for p in [args.output / "predictions.jsonl", args.output / "acquisition.jsonl"]},
                  "scripts": {str(p.relative_to(ROOT)): sha(p) for p in
                              [ROOT / "tools/prepare_proxy.py", Path(__file__), ROOT / "baselines/empirical_mean/model.py"]}}
    write_json(args.output / "run.json", provenance)


if __name__ == "__main__":
    main()
