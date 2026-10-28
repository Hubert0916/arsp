# Official formative feedback

- `runs.json`: per-record group-averaged metrics; no anonymous platform group IDs.
- `summary.json`: equal-record statistics over seven complete, distinct records,
  plus a separate summary of all eight distinct available ALC summaries.
- `ingestion.json`: input checksum, duplicate and incomplete-table audit.
- `raw/official_result.log`: original transcription, retained locally and ignored
  by Git. Its nine feedback headers do not support ten complete evaluations.

The supplied record identifies the method as ASRP with adaptive acquisition.
Submission IDs and archive hashes are absent, so archive identity cannot be
verified from the transcription. Identical block 7 repeats block 6; block 9 lacks
budget 31; sixteen additional rows lack a matching ALC summary. Those rows are
not assigned to a run or budget by inference.

Scores are averaged equally across subject–benchmark groups within a record,
then equally across records. Sample SD uses the number of records minus one.
Six-budget columns all use the same seven complete records. ALC uses weights
`0.1, 0.2, 0.2, 0.2, 0.2, 0.1`. Rounding checks allow `1.00001e-6` discrepancy
between a group summary and its weighted six-decimal budget scores.

From the repository root, regenerate report tables from these public aggregates:

```bash
uv run --locked python tools/update_official_report.py
```

To audit the original local transcription again:

```bash
uv run --locked python tools/parse_official_results.py
uv run --locked python tools/test_official_results.py
```

These commands reproduce aggregation, not the platform's private evaluations.
No matched official Empirical Mean or Random-acquisition feedback was supplied.
