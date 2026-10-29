# Official evaluation: pending

Official Codabench results for the current ASRP version (`prior-kernel-v2`,
without Rasch) have not been supplied. The report retains a placeholder for the
six label budgets and Brier ALC.

`status.json` identifies the pending model version and frozen inference files.
There are no released official scores or submission IDs for this version.

Regenerate the placeholder from the repository root:

```bash
uv run --locked python tools/update_official_report.py
```

After new feedback is available, the parser can audit its tables. Published
results must identify the evaluated model and correspond to the same inference
manifest before the report generator accepts them.
