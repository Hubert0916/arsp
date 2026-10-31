# Public formative simulation: Random acquisition

The current release compares ASRP and Empirical Mean using identical uniformly
sampled labels across 30 complete-group public draws. Each draw is capped at
1,000 subject–item pairs. Group selection, item pools, candidate order, and
recorded outcomes are pinned in `experiments/formative_manifest.json`.

See the [protocol](../../experiments/formative_protocol.md). This is a public-data
simulation with frozen fitting/calibration overlap, not a reproduction of private
Codabench draws.

- `predictions.jsonl.gz`: every evaluation response at all six budgets for full
  ASRP, Empirical Mean, and diagnostic component removals under Random acquisition.
- `acquisition.jsonl.gz`: decisions from the packaged Random hook, checked against
  the reference uniform-streaming rule at every candidate.
- `summary.json`, `draws.json`, `by_group.json`: equal-group/equal-draw scores.
- `run.json`: pinned inputs, inference manifest, scripts, and output checksums.
- `reproduction_check.json`: fresh inference compared with the previous release's
  Random records, including all probabilities and acquisition decisions.

Audit and regenerate report inputs:

```bash
uv run --locked python tools/score_draws.py results/formative
uv run --locked python tools/update_formative_report.py
uv run --locked python report/make_proxy_plot.py
```

Run `tools/reproduce.py` to reconstruct the pinned data, rerun the current Random
submission, and compare all prediction rows and acquisition decisions. Use
`--cache-only` with downloaded data or `--prepared` with verified prepared data.

Historical paired-policy results are available in Git revision `ee9257a`.
Earlier fixed-subject experiments under `../proxy`, `../ablation`, and
`../rasch_*` are historical and are not averaged into the current tables.
