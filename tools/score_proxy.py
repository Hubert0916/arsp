"""Independently recompute group-balanced Brier metrics from per-item predictions."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from prepare_proxy import ROOT, sha, write_json


def scores(records, budgets, weights):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (row["source"], row["subject_id"], row["item_id"])
        if key in seen:
            raise ValueError("Duplicate evaluation item")
        seen.add(key)
        if row["y"] not in (0, 1):
            raise ValueError("Non-binary evaluation outcome")
        for method in ("asrp", "empirical_mean"):
            if len(row[method]) != len(budgets) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row[method]):
                raise ValueError("Invalid prediction array")
        groups[key[:2]].append(row)
    if not groups:
        raise ValueError("No evaluation rows")
    result = {"status": "public_proxy", "n_sources": len({k[0] for k in groups}),
              "n_groups": len(groups), "n_rows": len(records), "budgets": budgets}
    per_group = []
    for (source, subject), rows in sorted(groups.items()):
        record = {"source": source, "subject_id": subject, "n_rows": len(rows)}
        for method in ("asrp", "empirical_mean"):
            brier = [math.fsum((r[method][i]-r["y"])**2 for r in rows)/len(rows) for i in range(len(budgets))]
            record[method+"_brier"] = brier
            record[method+"_alc"] = math.fsum(w*b for w, b in zip(weights, brier))
        per_group.append(record)
    for method in ("asrp", "empirical_mean"):
        result[method+"_brier"] = [math.fsum(g[method+"_brier"][i] for g in per_group)/len(per_group) for i in range(len(budgets))]
        result[method+"_alc"] = math.fsum(g[method+"_alc"] for g in per_group)/len(per_group)
    return result, per_group


def audit(records, acquisitions, manifest):
    expected = {(g["source"], g["subject_id"], r["item_id"]): r["response_id"]
                for g in manifest["groups"] for r in g["evaluation"]}
    actual = {(r["source"], r["subject_id"], r["item_id"]): r["response_id"] for r in records}
    if actual != expected:
        raise ValueError("Predictions do not cover the exact published evaluation split")
    trajectories = {(a["source"], a["subject_id"]): a for a in acquisitions}
    if len(trajectories) != len(manifest["groups"]) or len(acquisitions) != len(trajectories):
        raise ValueError("Missing or repeated acquisition trajectory")
    for group in manifest["groups"]:
        trajectory = trajectories[(group["source"], group["subject_id"])]
        pool = [r["item_id"] for r in group["acquisition"]]
        selected = trajectory["acquired_item_ids"]
        if len(selected) != 31 or len(set(selected)) != 31 or not set(selected) <= set(pool):
            raise ValueError("Invalid acquired items")
        decisions = trajectory["decisions"]
        if [d["position"] for d in decisions] != list(range(len(decisions))):
            raise ValueError("Acquisition stream out of order")
        if any(d["item_id"] != pool[d["position"]] for d in decisions):
            raise ValueError("Decision references the wrong candidate")
        if [d["item_id"] for d in decisions if d["acquired"]] != selected:
            raise ValueError("Selection and decision log disagree")
        labels = trajectory["labels"]
        if len(labels) != 31 or any(y not in (0, 1) for y in labels):
            raise ValueError("Invalid acquired outcomes")
    for row in records:
        labels = trajectories[(row["source"], row["subject_id"])]["labels"]
        expected_mean = [sum(labels[:n])/n if n else .5 for n in manifest["config"]["budgets"]]
        if row["empirical_mean"] != expected_mean:
            raise ValueError("Empirical Mean did not receive matched evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "results/reproduced")
    parser.add_argument("--compare", type=Path, help="Compare every prediction and acquisition with a reference run")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments/proxy_manifest.json").read_text())
    config = manifest["config"]
    run = json.loads((args.directory / "run.json").read_text())
    if run["config"] != config or run["split_manifest_sha256"] != sha(ROOT / "experiments/proxy_manifest.json"):
        raise ValueError("Run does not match the published protocol/split")
    if run["data_sha256"] != manifest["prepared_sha256"]:
        raise ValueError("Run used different prepared inputs")
    if run["n_groups"] != len(manifest["groups"]):
        raise ValueError("Incomplete run")
    for filename, expected in run["files"].items():
        if sha(args.directory / filename) != expected:
            raise ValueError(f"Result checksum mismatch: {filename}")
    records = [json.loads(line) for line in (args.directory / "predictions.jsonl").read_text().splitlines()]
    acquisitions = [json.loads(line) for line in (args.directory / "acquisition.jsonl").read_text().splitlines()]
    audit(records, acquisitions, manifest)
    result, groups = scores(records, config["budgets"], config["alc_weights"])
    result.update(seed=config["seed"], dataset_revision=config["revision"],
                  split_manifest_sha256=sha(ROOT / "experiments/proxy_manifest.json"))
    write_json(args.directory / "summary.json", result)
    write_json(args.directory / "by_source.json", groups)
    if args.compare:
        reference = [json.loads(line) for line in (args.compare / "predictions.jsonl").read_text().splitlines()]
        if len(reference) != len(records):
            raise ValueError("Reference row count differs")
        for a, b in zip(records, reference):
            for key in ("source", "subject_id", "item_id", "response_id", "y"):
                if a[key] != b[key]:
                    raise ValueError(f"Reference identity differs: {key}")
            for method in ("asrp", "empirical_mean"):
                if any(abs(p-q) > 1e-7 for p, q in zip(a[method], b[method])):
                    raise ValueError("Predictions differ by more than 1e-7")
        if (args.directory / "acquisition.jsonl").read_bytes() != (args.compare / "acquisition.jsonl").read_bytes():
            raise ValueError("Reference acquisition differs")
        print("All predictions match within 1e-7; acquisition trajectories match exactly.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
