"""Download pinned public tables and construct a deterministic, auditable proxy."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
          "reasoning_effort", "harness_version", "subject_features_extra")


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(config, cache_only=False):
    con = duckdb.connect(config={"threads": 2, "memory_limit": "2GB"})
    groups, files, selections = [], {}, []
    for source in config["sources"]:
        tables = {}
        for name in ("benchmarks", "subjects", "items", "responses"):
            filename = f"{source}/formatted_tables/{name}.parquet"
            path = hf_hub_download(config["dataset"], filename, repo_type="dataset",
                                   revision=config["revision"], local_files_only=cache_only)
            files[filename] = {"sha256": sha(path), "bytes": Path(path).stat().st_size}
            tables[name] = path
        def rows(sql, parameters):
            result = con.execute(sql, parameters)
            return [dict(zip([c[0] for c in result.description], row)) for row in result.fetchall()]
        metadata = rows("SELECT * FROM read_parquet(?)", [tables["benchmarks"]])
        if not metadata or any(r["response_type"] != "binary" for r in metadata):
            raise ValueError(f"{source}: expected declared binary responses")
        eligible = rows("""SELECT subject_id, count(DISTINCT item_id) AS n
            FROM read_parquet(?) WHERE response IN (0,1)
            GROUP BY subject_id HAVING n >= ?""", [tables["responses"], config["items_per_group"]])
        if not eligible:
            raise ValueError(f"{source}: no eligible subject; refusing silent source removal")
        sid = min(eligible, key=lambda r: digest(config["seed"], source, r["subject_id"]))["subject_id"]
        observed = rows("""SELECT * FROM read_parquet(?) WHERE subject_id=? AND response IN (0,1)
            QUALIFY row_number() OVER (PARTITION BY item_id ORDER BY trial, response_id)=1""",
                        [tables["responses"], sid])
        selected = sorted(observed, key=lambda r: digest(config["seed"], source, sid, r["item_id"]))[:config["items_per_group"]]
        ids = [r["item_id"] for r in selected]
        # Stream content columns: HTML-heavy sources can have multi-GB row groups.
        items = []
        for batch in pq.ParquetFile(tables["items"]).iter_batches(
                batch_size=128, columns=["item_id", "content", "item_features"]):
            items.extend(batch.filter(pc.is_in(batch.column("item_id"), value_set=pa.array(ids))).to_pylist())
        item_map = {r["item_id"]: r for r in items}
        if len(items) != len(ids) or len(item_map) != len(ids):
            raise ValueError(f"{source}: missing or duplicate item records")
        subjects = rows("SELECT * FROM read_parquet(?) WHERE subject_id=?", [tables["subjects"], sid])
        if len(subjects) != 1:
            raise ValueError(f"{source}: expected exactly one subject record")
        subject = {k: str(subjects[0][k] or "") for k in FIELDS}
        alias = "benchmark_" + digest(config["seed"], source)[:16]
        group = {"source": source, "subject_id": sid, "subject_alias": "subject_" + digest(source, sid)[:16],
                 "benchmark_alias": alias, "subject": subject, "acquisition": [], "evaluation": []}
        selection = {k: group[k] for k in ("source", "subject_id", "subject_alias", "benchmark_alias")}
        for index, record in enumerate(selected):
            item = item_map[record["item_id"]]
            part = "acquisition" if index < config["acquisition_items"] else "evaluation"
            entry = {"item_id": record["item_id"], "response_id": record["response_id"],
                     "label": int(record["response"]),
                     "input": [subject, {"item_content": item["content"] or "", "item_features": item["item_features"] or "",
                                         "interactors": record["interactors"] or "", "benchmark_id": alias}]}
            group[part].append(entry)
            selection.setdefault(part, []).append({k: entry[k] for k in ("item_id", "response_id")})
        groups.append(group)
        selections.append(selection)
        print(f"Prepared {len(groups)}/{len(config['sources'])}: {source}", flush=True)
    con.close()
    return groups, {"config": config, "files": files, "groups": selections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/proxy_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/proxy.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/proxy_manifest.json")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--write-manifest", action="store_true", help="Create a new split manifest instead of verifying the published one")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    groups, manifest = prepare(config, args.cache_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(g, ensure_ascii=False, separators=(",", ":")) + "\n" for g in groups)
    manifest["prepared_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    if args.write_manifest:
        write_json(args.manifest, manifest)
    elif manifest != json.loads(args.manifest.read_text()):
        raise ValueError("Data/split differs from the published manifest")
    args.output.write_text(payload)
    print(f"Wrote {args.output}; SHA256={manifest['prepared_sha256']}")


if __name__ == "__main__":
    main()
