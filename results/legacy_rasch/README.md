# Earlier model: initial component ablations

These archived predictions use the **earlier ASRP version containing Rasch**,
not the current runtime. They cover seeds 42/43/44, the same 92 source groups,
and matched Adaptive evidence. `full` and `no_rasch` supply the initial rows of
the removal comparison; the other variants also refer to the earlier model.

- `predictions.jsonl.gz`: earlier-model per-item component predictions.
- `summary.json` and `by_source.json`: equal-group, equal-seed scores.
- `run.json`: original inputs and checksums at the paths used by that version.

To rerun this historical experiment, use a separate checkout of Git revision
`f9afbd258cc1f47ef158487261c79db14a48eafa` and follow its README. That revision
contains the original `tools/evaluate_ablation.py`, model files, protocol, and
`results/ablation` directory. No historical model weights are required by the
current submission.

The additional seeds and removal decision are in
[`../rasch_confirmation`](../rasch_confirmation). Current-version component
results are in [`../proxy`](../proxy).
