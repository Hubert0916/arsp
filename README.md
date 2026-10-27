# ASRP: Adaptive Support-Based Response Prediction

**Yuhan Chiu — National Taiwan University**

ASRP predicts subject–item response probabilities at label budgets **0, 1, 3,
7, 15, 31**. It combines a prior-informed Rasch hybrid, a support-conditioned
Kernel Probe, topic calibration, and `adaptive_4_warm` label acquisition.
The repository URL uses `arsp`; the method is **ASRP**.

## Repository layout

```text
src/asrp/                   Prediction and acquisition source code
artifacts/asrp/             Frozen parameters, settings, and MiniLM encoders
artifacts/manifest.sha256.json  Integrity checks for inference files
baselines/empirical_mean/   Official Empirical Mean predictor and license
report/                    Paper source, PDF, figures, and recorded results
tools/                     Submission builder, smoke test, integrity check
docs/validation.md         Release validation record
pyproject.toml + uv.lock   Reproducible CPU environment
dist/                      Generated submission ZIPs (not tracked)
```

The official starting kit, historical experiments, research papers, intermediate
LaTeX files, and generated ZIPs are not included in the public source tree.

## Setup and build

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/Hubert0916/arsp.git
cd arsp
uv sync --locked
uv run python tools/verify_artifacts.py
uv run python tools/build_submission.py asrp
uv run python tools/smoke_test.py dist/asrp.zip
```

The builder combines `src/asrp/` and `artifacts/asrp/` into a ZIP with `model.py`
at its root. Run the packaged version rather than importing source files directly:
the inference loaders expect model files alongside the source in the submission.
Both MiniLM encoders are bundled (about 173 MiB combined); inference does not
fetch models or call external APIs. They are ordinary Git files, not LFS pointers.

Build and check the official Empirical Mean baseline:

```bash
uv run python tools/build_submission.py empirical_mean
uv run python tools/smoke_test.py dist/empirical_mean.zip
```

Upload the appropriate ZIP from `dist/`. ASRP includes `labeling.py`; Empirical
Mean omits it and uses the platform's default random acquisition policy.
The included smoke test validates synthetic predictions and acquisition, not
competition performance. For the **official submission checker**, download the
starting kit from [the competition page](https://aims.stanford.edu/competition)
and follow its instructions. The kit is maintained upstream and is not vendored here.

## Report and local results

[Technical report](report/main.pdf) · [LaTeX source](report/main.tex)

Recorded **local public-proxy** results (142 sources; 5,680 evaluated rows):

| Method | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASRP | 0.2245 | 0.2084 | 0.2005 | 0.1914 | 0.1854 | 0.1766 | 0.1973 |
| Empirical Mean | 0.2500 | 0.3391 | 0.2501 | 0.2119 | 0.1926 | 0.1850 | 0.2422 |

These are recorded results, not a new evaluation of this release or Codabench
scores. Both predictors used identical acquisition prefixes in this comparison.
Public calibration and proxy evaluation are not fully independent.

```bash
uv run python report/make_proxy_plot.py
# Requires latexmk, BibTeX, and TikZ installed locally:
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
```

## Reproducibility scope

Source, frozen artifacts, environment lockfile, integrity manifest, packaging,
and smoke checks support **running the frozen predictor**. The original complete
training, calibration-fitting, and final proxy-evaluation scripts were no longer
present in the retained workspace at release time. This repository does not
claim to reproduce training from scratch or independently verify the final
ASRP scores. Replotting `report/results_proxy.json` is not re-evaluation.
Historical diagnostic rows are retained locally and excluded from this release
because their ASRP predictions belong to an earlier version.

See [validation notes](docs/validation.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
Competition-specific artifacts derive from the permitted public
[measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
release. Codex assisted with development and report preparation; it is not an
online predictor.
