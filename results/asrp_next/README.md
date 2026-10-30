# No-Rasch candidate evaluation

**Decision: does not pass the predefined local validation gates; retain released ASRP.**

One global configuration was selected from 360 development settings, using 30 previously released draws and both acquisition policies. It was then locked before scoring 40 broad validation draws and 20 seven-benchmark-pool stress draws. The validation draws contain 176 distinct groups excluded from the previous proxy and formative development groups.

This is a public approximation of the documented formative protocol. The private coordinator and private pool are unavailable. Frozen prior/calibration training still overlaps the public release; the new groups are held out from candidate selection, not from all offline fitting.

## Selected configuration

```json
{
  "prior_shrink": 1.0,
  "prior_center": 0.5,
  "concentrations": [
    4.0
  ],
  "masses": [
    1.0
  ],
  "residual_scale": 1.0,
  "combine": "rebase",
  "topic_decay": 8.0
}
```

For the selected fixed-strength configuration, the prediction is `clip((s + 4*q)/(n + 4) + w_n*r + 8/(8+n)*delta, 0, 1)`. `q` is the frozen metadata prior, `r` is the support-conditioned kernel residual, and `delta` is the frozen topic correction. The kernel weights remain `[0, 0, 0.2, 0.4, 1, 1]`. The candidate retains the metadata-smoothed estimate at high budgets instead of replacing it with the original kernel intercept. Evidence-dependent concentration mixing and extra prior shrinkage were explored but are not used by this selected configuration.

## Matched validation ALC

| Pool | Acquisition | Released ASRP | Plain no-Rasch | Candidate | Empirical Mean |
|---|---|---:|---:|---:|---:|
| broad | random | 0.196678 | 0.197096 | 0.196633 | 0.233154 |
| broad | adaptive | 0.196306 | 0.196504 | 0.196024 | 0.230033 |
| small_pool | random | 0.184980 | 0.177179 | 0.175521 | 0.210842 |
| small_pool | adaptive | 0.184915 | 0.177058 | 0.175713 | 0.208124 |

All methods receive exactly the same acquired labels within each draw and acquisition policy. Lower is better. The two pools are reported separately.

## Adaptive: six budgets

| Pool | Method | 0 | 1 | 3 | 7 | 15 | 31 | ALC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| broad | Released ASRP (Rasch) | 0.222088 | 0.208461 | 0.198511 | 0.191925 | 0.183299 | 0.176585 | 0.196306 |
| broad | Plain no-Rasch | 0.224087 | 0.209210 | 0.198130 | 0.191545 | 0.183299 | 0.176585 | 0.196504 |
| broad | Refined no-Rasch | 0.224087 | 0.209257 | 0.198467 | 0.190760 | 0.181758 | 0.175664 | 0.196024 |
| small_pool | Released ASRP (Rasch) | 0.235078 | 0.211963 | 0.185690 | 0.166708 | 0.165043 | 0.155267 | 0.184915 |
| small_pool | Plain no-Rasch | 0.210450 | 0.194826 | 0.178361 | 0.164204 | 0.165043 | 0.155267 | 0.177058 |
| small_pool | Refined no-Rasch | 0.210450 | 0.195029 | 0.177578 | 0.162871 | 0.161206 | 0.153315 | 0.175713 |

## Adaptive: paired differences and ablations

| Pool | Comparison | Mean ALC difference | Paired bootstrap 95% interval |
|---|---|---:|---|
| broad | Candidate minus released ASRP | -0.000283 | [-0.002789, +0.002404] |
| broad | no_kernel minus candidate | +0.001344 | [+0.000476, +0.002198] |
| broad | no_topic minus candidate | +0.000329 | [-0.000662, +0.001365] |
| broad | neutral_metadata_prior minus candidate | +0.006186 | [+0.001450, +0.010756] |
| broad | original_kernel_blend minus candidate | +0.000176 | [-0.000404, +0.000728] |
| broad | no_topic_decay minus candidate | +0.000360 | [-0.000044, +0.000750] |
| small_pool | Candidate minus released ASRP | -0.009202 | [-0.011323, -0.007321] |
| small_pool | no_kernel minus candidate | +0.000629 | [+0.000010, +0.001240] |
| small_pool | no_topic minus candidate | +0.000600 | [+0.000052, +0.001052] |
| small_pool | neutral_metadata_prior minus candidate | +0.013140 | [+0.010160, +0.016213] |
| small_pool | original_kernel_blend minus candidate | +0.001191 | [+0.000583, +0.001813] |
| small_pool | no_topic_decay minus candidate | +0.000212 | [-0.000032, +0.000482] |

`no_kernel` removes the kernel residual; `no_topic` removes the complete topic correction; `neutral_metadata_prior` replaces the metadata prior with 0.5; `original_kernel_blend` restores the previous kernel-intercept blend while retaining candidate topic decay; `no_topic_decay` keeps topic correction at its original strength.

A negative candidate-minus-reference value favors the candidate. A positive ablation-minus-candidate value favors retaining the component. Intervals resample draws within the fixed evaluation pool; draws can reuse groups. These intervals do not establish generalization to a new private pool.

## Acceptance gates

- PASS: broad/random: mean improves full
- PASS: broad/random: mean improves no_rasch
- PASS: broad/adaptive: mean improves full
- PASS: broad/adaptive: mean improves no_rasch
- FAIL: broad: paired 95% upper below zero versus full
- PASS: broad: >=60% draw wins versus full
- PASS: broad: removing no_kernel does not improve mean
- PASS: broad: removing no_topic does not improve mean
- PASS: broad: removing neutral_metadata_prior does not improve mean
- PASS: broad: removing original_kernel_blend does not improve mean
- PASS: broad: removing no_topic_decay does not improve mean
- PASS: small_pool/random: mean improves full
- PASS: small_pool/random: mean improves no_rasch
- PASS: small_pool/adaptive: mean improves full
- PASS: small_pool/adaptive: mean improves no_rasch
- PASS: small_pool: paired 95% upper below zero versus full
- PASS: small_pool: >=60% draw wins versus full
- PASS: small_pool: removing no_kernel does not improve mean
- PASS: small_pool: removing no_topic does not improve mean
- PASS: small_pool: removing neutral_metadata_prior does not improve mean
- PASS: small_pool: removing original_kernel_blend does not improve mean
- PASS: small_pool: removing no_topic_decay does not improve mean

## Reproduction

See [protocol and commands](../../experiments/asrp_next/README.md). `development/search.json` contains all development settings; `validation/draws.json` contains every scored validation draw. `validation/summary.json` contains the locked configuration, hashes, comparisons, and gate decisions.

Original-curve reconstruction maximum Brier error: 5.38e-15. Plain no-Rasch reconstruction maximum probability error: 2.22e-16.

Standalone runtime: 384 prediction checks, 6 complete acquisition replays, maximum probability error 0. The standalone runtime contains no Rasch difficulty weights or subject-ability model.

An experimental ZIP can be built with `tools/package_prior_kernel.py`; `submission.json` records its checksum. It does not replace the released submission. `interface_checks.json` records the ZIP smoke test and the public starting-kit interface check; these checks are not competition performance estimates.
