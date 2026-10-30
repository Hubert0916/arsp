# ASRP public-proxy evaluation

**Historical protocol.** The current report uses the complete-group formative
simulation in [`../formative`](../formative). These fixed-subject results remain
available for comparison and use `tools/reproduce.py --legacy` for reproduction.

These predictions use the restored **with-Rasch** ASRP runtime, the same 92
selected source groups, seeds 42/43/44, and six budgets under Random and Adaptive
acquisition. See `experiments/proxy_config.json`.

- `predictions.jsonl`: ASRP and Empirical Mean probabilities for both policies.
- `acquisition.jsonl`: candidate decisions and acquired outcomes.
- `summary.json` and `by_source.json`: main scores and paired differences.
- `run.json`: frozen-input, script, and output checksums.
- `restoration_check.json`: full rerun of the restored package, reproducing all
  134,142 prediction records and 552 acquisition trajectories exactly.

From the repository root, regenerate scores with:

```bash
uv run --locked python tools/score_proxy.py results/proxy
```

`tools/reproduce.py --legacy` reruns inference and compares probabilities and acquisition
traces against this release. Component-removal predictions are in
[`../ablation`](../ablation), with additional Rasch draws in
[`../rasch_confirmation`](../rasch_confirmation). Remaining parameters and
calibration statistics are not refitted for these comparisons.
