# Rasch and count-prior interaction

**Historical fixed-subject experiment.** Current report results use the
[formative simulation](../formative); the scores below are not part of its tables.


A fixed 2×2 component ablation on all 92 public-proxy groups, seeds 42/43/44,
and budgets 0 and 1. Each condition uses exactly the published Adaptive support
and evaluation items. No weights or calibration statistics are refitted.

| Count prior | Rasch | B0 | B1 |
|---|---|---:|---:|
| Available | Enabled | 0.22633 | 0.21605 |
| Available | Disabled | 0.22770 | 0.21535 |
| Removed | Enabled | 0.24192 | 0.22675 |
| Removed | Disabled | 0.24409 | 0.22738 |

Removing the count prior sets `feature_prior=None`, invoking the native
`p_meta=mu_s` fallback. It does **not** remove visible metadata, the binary subject
prior, or topic corrections. Removing Rasch sets `lambdas` to zero. The kernel
has zero weight at both checkpoints.

Rasch's marginal contribution is more favorable without the count prior. Its
final-score effect also depends on the fixed topic calibration: before correction,
Rasch slightly worsens B0 with the normal count prior; without the count prior,
it improves raw B0 while raw B1 is almost tied. See `effects.json` and the raw
scores in `summary.json`. These are conditional component effects, not proof of
universal necessity, a held-out-source evaluation, or an explanation established
for private-platform results.

- `predictions.jsonl.gz`: all 67,071 evaluation records, four conditions, and
  both pre-calibration and final probabilities at the two budgets.
- `summary.json`, `by_source.json`, `effects.json`: equal-group/seed scores and interactions.
- `run.json`: checksums and fixed protocol.

`run.json` preserves the inference-script checksum. The scorer subsequently gained
an additional check against the original pre-calibration predictions;
`summary.json` records the scoring-script checksum separately.

The two normal-count-prior conditions reproduce the original ablation predictions
exactly. Native inference is also checked for every condition on the first target
of each group/seed/budget.

After preparing data with the main reproduction pipeline:

```bash
uv run --locked python tools/evaluate_rasch_prior.py --workers 4 --output results/check-prior
```

To audit the released predictions and regenerate report inputs:

```bash
uv run --locked python tools/evaluate_rasch_prior.py --score-only
uv run --locked python tools/update_rasch_prior_report.py
```
