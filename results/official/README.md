# Official formative feedback

- `runs.json`: per-record group-averaged metrics; no anonymous platform group IDs.
- `summary.json`: equal-record statistics over ten complete, distinct records.
- `ingestion.json`: restored-input checksum and duplicate audit.
- `provenance.json`: same-ZIP confirmation and transcription restoration steps.
- `submissions.json`: Codabench submission IDs and displayed scores from the supplied screenshot, matched uniquely to feedback records.
- `raw/official_result.log`: restored feedback; retained locally and ignored by Git.
- `raw/official_result_original_incomplete.log`: byte-preserved original copy.
- `raw/user_supplement_2026-10-28.log`: added material transcribed from the reply.

The submitter confirmed on 2026-10-28 that every evaluation used the same ASRP +
`adaptive_4_warm` ZIP. The final attachment contains ten complete, distinct feedback blocks,
including the corrected record at block 7. It supersedes the earlier pasted
copies. All ten records and all six budgets are included, with no imputed scores.
Submission IDs are supplied separately in the Codabench screenshot and mapped in
`submissions.json`; archive hashes are absent, and same-ZIP identity is
submitter-confirmed. Screenshot status is retained as displayed, including one
Running entry with a score; all ten records have complete scoring feedback. The attachment is preserved byte-for-byte locally as
`raw/user_full_feedback_2026-10-28.txt`; only a single header typo was normalized
in the parser input. The prior restored copy is retained as
`raw/official_result_previous_restored.log`.

Scores are averaged equally across subject–benchmark groups within a record,
then equally across records. Sample SD uses the number of records minus one.
Six-budget columns all use the same ten records. ALC uses weights
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
