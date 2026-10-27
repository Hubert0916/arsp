# ASRP: Adaptive Support-Based Response Prediction

**Yuhan Chiu — National Taiwan University**

ASRP predicts whether an AI subject will answer an item correctly under six label
budgets: **0, 1, 3, 7, 15, 31**. It combines a public-data prior and amortized Rasch
hybrid at low budgets with a support-conditioned Kernel Probe as evidence grows.
The final predictor includes topic residual calibration, a source guard, and the
`adaptive_4_warm` streaming acquisition policy.

This repository contains the competition inference implementation, frozen
artifacts, the official Empirical Mean comparator, local submission checks,
and the technical report. The repository URL uses `arsp`; the method is **ASRP**.

## Setup and validation

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/Hubert0916/arsp.git
cd arsp
uv sync --locked
uv run python tools/verify_artifacts.py
uv run python starting_kit/tools/run_smoke_test.py project/asrp
```

The environment uses CPU PyTorch. Both MiniLM model files are bundled (about
173 MiB combined), so inference does not download models or call external APIs.
They are ordinary Git files, not Git LFS pointers. The small competition-specific
JSON/NPZ artifacts are also bundled.

## Build a Codabench submission

```bash
uv run python tools/build_submission.py asrp
uv run python starting_kit/tools/check_submission_zip.py dist/asrp.zip

uv run python tools/build_submission.py empirical_mean
uv run python starting_kit/tools/check_submission_zip.py dist/empirical_mean.zip
uv run python starting_kit/tools/run_smoke_test.py baselines/empirical_mean
```

Upload the appropriate ZIP from `dist/`. Files such as `model.py` are placed at
its root. ASRP supplies `labeling.py`; Empirical Mean omits it and uses the
platform's default random acquisition policy. A smoke test checks the interface;
it is not a competition performance evaluation. See [validation notes](VALIDATION.md)
for the checks performed on this release.

## Report and local results

See [technical report](report/main.pdf) and [LaTeX source](report/main.tex).
The report records the following **local public proxy**, with 142 sources and
5,680 evaluated rows:

| Method | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASRP | 0.2245 | 0.2084 | 0.2005 | 0.1914 | 0.1854 | 0.1766 | 0.1973 |
| Empirical Mean | 0.2500 | 0.3391 | 0.2501 | 0.2119 | 0.1926 | 0.1850 | 0.2422 |

These are recorded local results, not official Codabench results or a new run of
this release. The comparison uses identical acquisition prefixes. It therefore
differs from comparing the two standalone submissions' acquisition policies.
Public calibration and proxy evaluation are not fully independent.

Regenerate the figure from recorded aggregate values:

```bash
uv run python report/make_proxy_plot.py
# Requires a local TeX installation with latexmk, BibTeX, and TikZ:
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
```

## Reproducibility scope

- **Inference and packaging:** source, frozen parameters, encoders, environment
  lockfile, checksums, ZIP builder, and interface checks are included.
- **Empirical Mean reconstruction:**
  `uv run python report/recompute_proxy_baselines.py` recovers acquired success
  counts from saved public diagnostic rows and recomputes the baseline curve.
- **Historical diagnostics:** `analysis/analysis_source_topic.json` and
  `analysis/source_topic_summary.md` belong to an earlier ASRP version. Their
  ASRP predictions do **not** reproduce the final aggregate ASRP curve in the
  report. They are included for provenance and Empirical Mean reconstruction.
- **Not included:** the original complete training, calibration-fitting, and
  final proxy-evaluation scripts were no longer present in the retained workspace
  at release time. This release supports running the frozen predictor; it does
  not claim to reproduce training from scratch or independently verify the
  report's final ASRP scores. Replotting aggregates is not re-evaluation.

## Layout

- `project/asrp/`: inference, acquisition, encoder files, and frozen parameters.
- `baselines/empirical_mean/`: official baseline and its license.
- `starting_kit/`: official local checks and synthetic example data.
- `tools/`: packaging and artifact verification.
- `analysis/`: historical public-data diagnostics, as qualified above.
- `report/`: report source, PDF, figures, and recorded result aggregates.

Training data provenance is the permitted public
[measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
release. See [third-party notices](THIRD_PARTY_NOTICES.md). Codex assisted with
implementation and report preparation; it is not an online predictor.
