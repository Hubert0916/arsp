# Official Empirical Mean baseline

Source: https://github.com/aims-foundations/paiec_baseline
The upstream MIT license is retained in `LICENSE`.

For each subject–benchmark pair, predict the fraction of successful acquired
responses, or 0.5 when no matching labels are available. There is no smoothing.
The predictor uses the full subject attributes and anonymous `benchmark_id` to
match evidence. It does not model differences between query items.

From this repository's root:

```bash
uv run python tools/build_submission.py empirical_mean
uv run python tools/smoke_test.py dist/empirical_mean.zip
```

The ZIP has `model.py` at its root and omits `labeling.py`, so Codabench uses its
default random acquisition policy. No model files or API credentials are required.
