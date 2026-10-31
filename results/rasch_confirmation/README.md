# Paired Rasch retention/removal comparison

**Historical commands:** use Git revision `ee9257a` for this paired-policy experiment. The current release supports Random acquisition only.

**Historical fixed-subject experiment.** Current report results use the
[formative simulation](../formative); the scores below are not part of its tables.


The initial comparison used seeds 42/43/44. This additional check uses seeds
45/46/47 on the **same 92 source groups**, with identical acquired labels and
evaluation items within each pair. It tests draw sensitivity, not new-source
transfer. Remaining frozen parameters and topic corrections are unchanged.

| Adaptive draw set | With Rasch ALC | Without Rasch ALC |
|---|---:|---:|
| Initial seeds 42–44 | 0.19871 | 0.19853 |
| Additional seeds 45–47 | 0.19880 | 0.19892 |
| All six seeds | 0.19875 | 0.19872 |

The budget effects offset: Rasch improves mean B0 and B7, slightly worsens B1
and B3, and has zero mixture weight at B15 and B31. `budget_analysis.json` gives
the exact weighted contribution of each checkpoint to ALC. `budget_by_source.json`
contains every source's six-seed mean difference; effects vary across sources.
The overall near-zero ALC difference does not imply unchanged predictions.

The additional [factorial prior ablation](../rasch_prior_ablation) tests how
Rasch's contribution depends on the retained count prior and topic calibration.
The evidence characterizes its conditional role in the frozen hybrid.

`predictions.jsonl.gz` contains both variants for the additional draws;
`acquisition.jsonl.gz` records their shared evidence. `summary.json`,
`by_source.json`, and `run.json` contain scores and provenance. Initial results
are in [`../ablation`](../ablation).

After preparing `data/proxy.jsonl`, reproduce from the repository root:

```bash
uv run --locked python tools/confirm_rasch_removal.py --workers 4
uv run --locked python tools/update_rasch_report.py
```

The tool extracts the earlier runtime from the pinned Git revision specified in
`experiments/rasch_confirmation.json`; it does not restore that runtime into the
current source tree. Full Git history is required. Use `--score-only` to audit
the saved predictions without inference.
