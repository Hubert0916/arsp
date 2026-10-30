# Rasch effect and implementation audit

The public comparison does **not** establish that removing Rasch improves ASRP.
No-Rasch minus full-ASRP ALC is -0.000239; a paired-draw percentile bootstrap
interval is [-0.002556, 0.002021]. It spans zero and is conditional on the fixed
public snapshot, not a test of private-data generalization or equivalence.

- `local_effects.json`: 100,000 paired-draw bootstrap replicates, fixed seed,
  budget effects, draw range, and weighted positive/negative contributions.
- `group_contributions.json`: all group-occurrence effects with the report's
  equal-group/equal-draw weights. Positive contributions sum to +0.003556 and
  negative contributions to -0.003795, explaining the small net effect.
- `version_check.json`: archived no-Rasch code (`2f85fdb`) versus the original
  implementation with Rasch weights set to zero. All 2,112 checked predictions
  agree exactly: first/last distinct inputs of each group, every draw, policy,
  and budget. This is not a check of every input or an unavailable platform ZIP.
- `context_check.json`: hold focal queries and labels fixed, adding only another
  subject's evidence within the same benchmark. This changes kernel-dependent
  predictions even at budgets 15/31, where Rasch has zero weight.

The three context examples were selected by lexical source order and group IDs,
without using scores. Acquisition is not rerun jointly. The tests demonstrate
context sensitivity; they do not identify the cause of any private-platform gap.
In the Babilong example, budget-15 Brier changes from 0.28805 to 0.23845 despite
identical focal labels and queries. Main public draws have no repeated benchmarks,
so these diagnostics are reported separately, not pooled into performance scores.

After the normal public-data preparation:

```bash
uv run --locked python tools/audit_rasch_effects.py
uv run --locked python tools/check_rasch_versions.py --workers 4
uv run --locked python tools/check_shared_context.py
uv run --locked python tools/update_formative_report.py
```

The version check requires Git history containing `2f85fdb`. No inference code,
frozen weights, or submitted model was changed by this audit. Private feedback
and version-comparison details remain local; the paper's official score table
continues to show the ten evaluations of the with-Rasch submission.
