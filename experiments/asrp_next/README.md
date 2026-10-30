# No-Rasch development experiment

This experiment tests a replacement predictor. The released ASRP source,
artifact manifest, report, and submission remain unchanged until validation
passes. `results/asrp_next/validation/summary.json` records the acceptance gates
and the resulting decision when the evaluation is complete.

## Predictor

The candidate keeps the frozen subject/metadata prior, support-conditioned
kernel, topic correction, and `adaptive_4_warm` acquisition. It removes the
Rasch prediction branch, difficulty MLP, and MAP ability update.

For a query prior `q`, the mean prior `u` of the acquired items, `n` acquired
labels, and `s` successes, development explores:

1. Shrinking `q` and `u` toward one global center.
2. Averaging `(s + c*q)/(n + c)` over a small fixed set of concentrations `c`.
   Component weights may adapt to the acquired outcomes using a beta-binomial
   evidence term centered on `u`.
3. Adding a damped kernel residual to this estimate, or blending it with the
   original kernel prediction.
4. Keeping the frozen topic correction or reducing it as evidence accumulates.

The evidence-weighted smoother is an approximation for heterogeneous items;
it is not presented as an exact generative posterior over all item responses.
One configuration is selected globally, without source-specific selection.
The acquisition rule is unchanged, so predictor comparisons are paired on the
same selected labels and evaluation responses.

## Development and validation

- Development uses the released 30 formative-style draws, both Random and
  Adaptive acquisition, and their original complete groups.
- Validation was fixed before scoring candidates: 40 broad draws and 20
  draws from a fixed seven-benchmark pool with groups of 80–160 items.
- The validation pool excludes the union of groups in the previous formative
  and proxy experiments. The selected validation draws contain 176 distinct
  groups. The smaller pool tests shared evidence from multiple subjects of
  the same benchmark.
- This is held-out validation for candidate selection. Frozen training and
  calibration artifacts still use the same public release; it is not a
  completely training-disjoint or private evaluation.
- Both cohorts retain the existing evaluator's 1,000-pair draw cap, complete
  eligible groups, fixed half acquisition/half evaluation pools, all evaluation
  responses, six budget barriers, and global revealed evidence. The small-pool
  restriction is an explicit stress test, not an asserted private sampling rule.

`protocol.json` fixes the seeds, excluded groups, selection rule, and gates.
`validation_manifest.json.gz` records the actual selected groups and responses;
the preparation command regenerates the uncompressed working manifest.
`locked_candidate.json` binds the chosen configuration to the development
results, implementation, and protocol before validation scores are inspected.

ALC is calculated per group from six Brier scores with weights
`[0.1, 0.2, 0.2, 0.2, 0.2, 0.1]`, averaged equally across groups in each draw,
then equally across draws. All repeated binary responses are scored exactly
through their counts and number of successes. Evaluation outcomes never enter
prediction or acquisition.

Acceptance requires improvement over both released ASRP and plain Rasch
removal in both policies/cohorts, a negative paired 95% bootstrap upper bound
versus released ASRP and at least 60% draw wins for Adaptive in each cohort,
and no improvement from removing any retained candidate component. Bootstrap
intervals describe these fixed public evaluation pools; they do not establish
private competition performance.

## Reproduce

Run from the repository root after installing dependencies and reproducing the
original formative data and frozen artifacts as described in the main README:

```bash
uv run --locked python tools/prepare_prior_kernel_validation.py
uv run --locked python tools/cache_prior_kernel.py --workers 8
uv run --locked python tools/cache_prior_kernel.py \
  --manifest experiments/asrp_next/validation_manifest.json \
  --data data/asrp_next_validation.jsonl \
  --output results/asrp_next/validation_cache --workers 8
uv run --locked python tools/search_prior_kernel.py
uv run --locked python tools/validate_prior_kernel.py
uv run --locked python tools/build_prior_kernel.py
uv run --locked python tools/check_prior_kernel_runtime.py
uv run --locked python tools/summarize_prior_kernel.py
uv run --locked python tools/package_prior_kernel.py
```

The last command creates `dist/asrp_no_rasch_candidate.zip` for experimental
platform testing. It requires completed validation and native runtime checks,
records whether all stability gates passed, and never replaces `dist/asrp.zip`.
Its existence is not a claim that the candidate passed every acceptance gate.

Validation must not be used to tune the selected configuration. If it fails,
the result is retained as a failed validation. A subsequent development round
would require a new confirmation set.

The released caches can also be rescored directly with
`tools/validate_prior_kernel.py` and summarized with
`tools/summarize_prior_kernel.py`, without running neural inference. To rebuild
an existing experimental runtime, pass a new `--output` directory to the builder
and the same directory as `--bundle` to the runtime check. An interrupted cache
can be resumed with `tools/resume_prior_kernel_cache.py`, preserving completed
arrays and replaying acquisition to recover missing traces.

The component cache checks original native predictions at every group and
checkpoint. The development scorer additionally checks all original Brier
curves against released results and requires the candidate family to reproduce
plain no-Rasch predictions to numerical precision. The independent runtime
check replays complete acquisition trajectories and checks the first and last
evaluation inputs of each group at every budget in one development draw and
one draw from each validation cohort.
