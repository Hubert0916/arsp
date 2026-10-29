# Paired Rasch removal check

The initial comparison used seeds 42/43/44. This additional check uses seeds
45/46/47 on the **same 92 source groups**, with identical acquired labels and
evaluation items within each pair. It tests draw sensitivity, not new-source
transfer. Remaining frozen parameters and topic corrections are unchanged.

| Adaptive draw set | With Rasch ALC | Without Rasch ALC |
|---|---:|---:|
| Initial seeds 42–44 | 0.19871 | 0.19853 |
| Additional seeds 45–47 | 0.19880 | 0.19892 |
| All six seeds | 0.19875 | 0.19872 |

The aggregate difference is negligible. Removal improves four of six seeds but
slightly worsens average zero-label Brier (0.22583 to 0.22696). We choose the
simpler prior/kernel architecture; this does not establish that Rasch is harmful
or uniformly unnecessary. The decision is recorded in `decision.json`.

`predictions.jsonl.gz` contains both variants for the additional draws;
`acquisition.jsonl.gz` records their shared evidence. `summary.json`,
`by_source.json`, and `run.json` contain scores and provenance. Initial results
are archived in [`../legacy_rasch`](../legacy_rasch).

After preparing `data/proxy.jsonl`, reproduce from the repository root:

```bash
uv run --locked python tools/confirm_rasch_removal.py --workers 4
uv run --locked python tools/update_rasch_report.py
```

The tool extracts the earlier runtime from the pinned Git revision specified in
`experiments/rasch_confirmation.json`; it does not restore that runtime into the
current source tree. Full Git history is required. Use `--score-only` to audit
the saved predictions without inference.
