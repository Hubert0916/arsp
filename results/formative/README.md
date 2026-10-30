# Public formative simulation

These results replace the report's fixed-subject proxy. Thirty predeclared draws
select complete groups under a 1,000-pair cap from the eligible public inventory.
Subjects can change between draws; each group's acquisition/evaluation pools,
candidate order, and recorded acquisition responses remain fixed.

See the [protocol](../../experiments/formative_protocol.md) for alignment with
published PAIEC rules and the local choices required because the private sampler
is unavailable. This is public-data evaluation with frozen fitting/calibration
overlap, not a reproduction of private test scores.

- `predictions.jsonl.gz`: every evaluation response under both policies, at six
  budgets, for ASRP, Empirical Mean, and three component removals.
- `acquisition.jsonl.gz`: candidate decisions and acquired responses; evidence
  grows globally with checkpoint barriers.
- `summary.json`, `draws.json`, `by_group.json`: equal-group/equal-draw scores.
- `rasch_by_group.json`: paired Rasch effects, averaging each group's repeated
  appearances before summarizing heterogeneity.
- `run.json`: data, frozen-inference, script and output checksums.
- `data_reproduction_check.json`: independent reconstruction of the pinned inputs
  and exact manifest comparison.
- `reproduction_check.json`: a fresh-process rerun of one complete draw, including
  both policies and all prediction variants.
- `independent_audit.json`: independent NumPy scoring and label verification for
  all released response records.

Audit and regenerate report values without rerunning inference:

```bash
uv run --locked python tools/score_draws.py results/formative
uv run --locked python tools/update_formative_report.py
uv run --locked python report/make_proxy_plot.py
```

Run the complete reproduction pipeline with `tools/reproduce.py`. It recreates
the pinned data, draws and inference, then compares all prediction rows and
acquisition decisions with these records. Use `--cache-only` with downloaded
data, or `--prepared` with verified `data/formative.jsonl`.

Earlier results in `../proxy`, `../ablation`, `../rasch_confirmation`, and
`../rasch_prior_ablation` use different protocols and are historical comparisons;
their scores are not averaged into this release.
