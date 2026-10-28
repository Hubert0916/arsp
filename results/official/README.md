# Official formative feedback

- `runs.json`: per-record group-averaged metrics; no anonymous platform group IDs.
- `summary.json`: equal-record statistics over nine complete, distinct records.
- `ingestion.json`: restored-input checksum and duplicate audit.
- `provenance.json`: same-ZIP confirmation and transcription restoration steps.
- `raw/official_result.log`: restored feedback; retained locally and ignored by Git.
- `raw/official_result_original_incomplete.log`: byte-preserved original copy.
- `raw/user_supplement_2026-10-28.log`: added material transcribed from the reply.

The submitter confirmed on 2026-10-28 that every evaluation used the same ASRP +
`adaptive_4_warm` ZIP. The completed transcription contains ten feedback blocks.
Block 7 exactly repeats block 6 and is counted once without separate submission
evidence, leaving nine complete, distinct records. All six budgets are available.
Missing sections were supplied by the submitter, not imputed. Submission IDs and
archive hashes are absent from the text; same-ZIP identity is submitter-confirmed.

Scores are averaged equally across subject–benchmark groups within a record,
then equally across records. Sample SD uses the number of records minus one.
Six-budget columns all use the same nine records. ALC uses weights
`0.1, 0.2, 0.2, 0.2, 0.2, 0.1`. Rounding checks allow `1.00001e-6` discrepancy
between a group summary and its weighted six-decimal budget scores.

From the repository root, regenerate report tables from these public aggregates:

```bash
uv run --locked python tools/update_official_report.py
```

To audit the restored local transcription again:

```bash
uv run --locked python tools/parse_official_results.py
uv run --locked python tools/test_official_results.py
```

These commands reproduce aggregation, not the platform's private evaluations.
No matched official Empirical Mean or Random-acquisition feedback was supplied.
