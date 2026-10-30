# ASRP: Adaptive Support-Based Response Prediction

**Yuhan Chiu — National Taiwan University**

ASRP predicts an AI system's probability of answering an item correctly with
0–31 revealed labels. It combines a Rasch ability–difficulty model, metadata-prior
smoothing, Kernel Probe, topic calibration, and adaptive label acquisition.

[Report](report/main.pdf) · [Method code](src/asrp) · [Report source and build instructions](report/README.md)

## Results

**Local public proxy:** 92 source groups, seeds 42/43/44, and six label budgets.
Scores average groups and seeds equally; lower is better.

<!-- RESULTS_START -->

| Predictor + acquisition | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empirical Mean + Random | 0.2500 | 0.3488 | 0.2518 | 0.2086 | 0.1910 | 0.1842 | 0.2435 |
| Empirical Mean + Adaptive | 0.2500 | 0.3594 | 0.2366 | 0.2003 | 0.1883 | 0.1837 | 0.2403 |
| ASRP + Random | 0.2263 | 0.2166 | 0.2075 | 0.1947 | 0.1845 | 0.1755 | 0.2008 |
| ASRP + Adaptive | 0.2263 | 0.2161 | 0.2043 | 0.1912 | 0.1819 | 0.1738 | 0.1987 |

<!-- RESULTS_END -->

Prediction methods share acquired labels within each policy. Public training and
proxy evaluation use the same data release, so this is not a source-held-out test.
See [protocol](experiments/proxy_config.json) and [full results](results/proxy).
The release provides reusable source selections, disjoint splits, per-item
predictions, and acquisition traces, with source-level comparisons in the
[scored summary](results/proxy/summary.json).

[Component ablations](results/ablation) remove Rasch, Kernel Probe, or topic
calibration with matched Adaptive labels. Across [six paired seeds](results/rasch_confirmation),
Rasch improves zero-label Brier (0.22583 vs 0.22696) but leaves mean ALC essentially
unchanged (0.19875 vs 0.19872). We retain the original with-Rasch submission.

**Official Codabench:** the ten with-Rasch evaluations have mean ALC **0.19107**
(SD 0.01105); the later 16 completed no-Rasch submissions average **0.20893**
(SD 0.02813). These are different random draws and dates, so this comparison
does not isolate Rasch's effect. [Records and aggregation](results/official).

## Reproduce local results

Install [uv](https://docs.astral.sh/uv/) and accept the access conditions for
[measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db), then run:

```bash
git clone https://github.com/Hubert0916/arsp.git
cd arsp
uv sync --locked
uv run hf auth login
uv run --locked python tools/reproduce.py
```

This downloads pinned data, uses bundled frozen weights, evaluates all four
combinations, and checks predictions against the published results. It reproduces
inference and scoring; original offline fitting scripts are not included. Outputs go
to `results/reproduced/`. Inference runs on CPU without external API calls.
Allow about 6 GB of disk plus the Python environment. Use `--workers 1` for fewer
workers, `--cache-only` for cached data, or `--prepared` to reuse prepared inputs.

To recalculate scores from published predictions without rerunning inference:

```bash
uv run --locked python tools/score_proxy.py results/proxy
uv run --locked python tools/update_ablation_report.py
uv run --locked python tools/confirm_rasch_removal.py --score-only
uv run --locked python tools/update_rasch_report.py
```

To rerun component inference and additional Rasch draws on the same sources:

```bash
uv run --locked python tools/evaluate_ablation.py --workers 4 --output results/check-ablation
uv run --locked python tools/confirm_rasch_removal.py --workers 4 --output results/check-rasch
```

## Build a submission

```bash
uv run --locked python tools/verify_artifacts.py
uv run --locked python tools/build_submission.py asrp
uv run --locked python tools/smoke_test.py dist/asrp.zip
```

Upload `dist/asrp.zip` to Codabench. To build the baseline, replace `asrp` with
`empirical_mean` in the build and smoke-test commands. The smoke test checks
packaging and execution; it does not measure benchmark performance.

## Files

- [`src/asrp/`](src/asrp): prediction and acquisition code.
- [`artifacts/`](artifacts/README.md): frozen models, settings, and reproduction scope.
- [`baselines/empirical_mean/`](baselines/empirical_mean): official baseline.
- [`experiments/`](experiments): protocol and pinned data/split manifest.
- [`results/`](results): public-proxy predictions, ablations, and official feedback.
- [`report/`](report): paper, figures, and LaTeX source.
- [`tools/`](tools): reproduction, scoring, and packaging scripts.

See [third-party notices](THIRD_PARTY_NOTICES.md). Codex assisted with development
and writing; it is not an online predictor.
