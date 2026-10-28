# Official formative feedback

- `runs.json`: per-record group-averaged metrics; no anonymous platform group IDs.
- `summary.json`: equal-record statistics over ten complete, distinct records.
- `ingestion.json`: input checksum and duplicate audit.
- `provenance.json`: same-ZIP confirmation and transcription history.
- `submissions.json`: Codabench submission IDs and displayed scores from the supplied screenshot, matched uniquely to feedback records.
- `raw/official_result.log`: feedback retained locally and ignored by Git.

We evaluated the same ASRP + `adaptive_4_warm` ZIP ten times. All records include
all six budgets, with no imputed scores. Submission IDs are matched uniquely to
the supplied screenshot by ALC within rounding tolerance. Same-ZIP identity is
author-confirmed; archive hashes were not supplied. Original transcriptions and
their restoration history are retained locally and documented in `provenance.json`.

Scores are averaged equally across subject–benchmark groups within a record,
then equally across records. Sample SD uses the number of records minus one.
Six-budget columns all use the same ten records. ALC uses weights
`0.1, 0.2, 0.2, 0.2, 0.2, 0.1`. Rounding checks allow `1.00001e-6` discrepancy
between a group summary and its weighted six-decimal budget scores.

From the repository root, regenerate report tables from these public aggregates:

```bash
uv run --locked python tools/update_official_report.py
```

When the local transcription is available, audit it with:

```bash
uv run --locked python tools/parse_official_results.py
uv run --locked python tools/test_official_results.py
```

These commands reproduce aggregation, not the platform's private evaluations.
No matched official Empirical Mean or Random-acquisition feedback was supplied.
