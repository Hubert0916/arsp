# Component ablations

**Historical commands:** use Git revision `ee9257a` for this paired-policy experiment. The current release supports Random acquisition only.

**Historical fixed-subject experiment.** Current report results use the
[formative simulation](../formative); the scores below are not part of its tables.


Inference-time removals of the Rasch term, Kernel Probe, and topic calibration
on all 92 public-proxy source groups and seeds 42/43/44. Each variant replays the
same published **Adaptive** acquisition trajectory and uses the same evaluation
items as full ASRP. The frozen parameters, remaining mixing weights, and
calibration statistics are retained without refitting or retuning.

- `predictions.jsonl.gz`: per-item probabilities at all six budgets for each variant.
- `summary.json`: equal-group, equal-seed means and paired ALC changes.
- `by_source.json`: source/seed/variant Brier curves and ALC.
- `run.json`: input, artifact-manifest, script, and output checksums.
- [`ablation_config.json`](../../experiments/ablation_config.json): exact removals.

After preparing `data/proxy.jsonl` with the main reproduction pipeline, rerun:

```bash
uv run --locked python tools/evaluate_ablation.py --workers 4 --output results/check-ablation
```

To audit the published predictions, recalculate scores, and rebuild the table:

```bash
uv run --locked python tools/update_ablation_report.py
```

The runner checks its shared component calculations against the native runtime
with the relevant component disabled at every group/seed/budget. Full ASRP must
also match every published Adaptive prediction within `1e-7`. These checks verify
that the comparison preserves the original evidence and full-model behavior.

The experiment measures the effect of removing a component from the frozen
system. It does not compare separately optimized models or establish how the
components behave on private competition data.
