# Official Codabench evaluations

The report presents ten evaluations of **ASRP with Rasch + adaptive acquisition**:
mean Brier ALC **0.19107**, sample SD **0.01105**. All six budget tables are
available. Group scores are averaged equally within each run, then runs equally.

- `runs.json`, `summary.json`: per-run aggregates and their verified summary.
- `submissions.json`: platform IDs matched to the rounded feedback ALC.
- `status.json`: local inference version and manifest checksum.
- `provenance.json`, `ingestion.json`: feedback origins and checks.

The author confirmed one ZIP across these ten evaluations. Current local inference
files match the pre-removal revision. The historical submitted ZIP hash is
unavailable. Original feedback is retained locally and excluded from this repository.

To verify the aggregates and regenerate report tables:

```bash
uv run --locked python tools/update_official_report.py
```

Private platform evaluations cannot be rerun using public data. Controlled
component comparisons use [`../formative`](../formative) and the diagnostic
checks in [`../rasch_audit`](../rasch_audit). Earlier protocols remain historical.
