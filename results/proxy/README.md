# Current ASRP public-proxy evaluation

The current prior/kernel model has no Rasch component. All results use the same
92 selected source groups, seeds 42/43/44, six budgets, and paired predictions
under Random and Adaptive acquisition. See `experiments/proxy_config.json`.

- `predictions.jsonl`: ASRP and Empirical Mean probabilities for both policies.
- `acquisition.jsonl`: candidate decisions and acquired outcomes.
- `summary.json` and `by_source.json`: main scores and paired differences.
- `component_predictions.jsonl.gz`: full/no-kernel/no-topic predictions using
  the same Adaptive evidence, captured in the main inference pass.
- `ablation_summary.json` and `ablation_by_source.json`: current component scores.
- `run.json`: frozen-input, script, and output checksums.

From the repository root, regenerate scores with:

```bash
uv run --locked python tools/score_proxy.py results/proxy
uv run --locked python tools/score_ablation.py results/proxy
```

`tools/reproduce.py` reruns inference and compares the resulting predictions and
acquisition traces to this release. Components are disabled only at inference;
remaining parameters and calibration statistics are not refitted.
