# ASRP: Adaptive Support-Based Response Prediction

**Yuhan Chiu — National Taiwan University**

ASRP predicts subject–item response probabilities at label budgets **0, 1, 3,
7, 15, 31**. It combines a prior-informed Rasch hybrid, a support-conditioned
Kernel Probe, topic calibration, and the `adaptive_4_warm` acquisition hook.
The repository URL uses `arsp`; the method is **ASRP**.

[Technical report](report/main.pdf) · [LaTeX source](report/main.tex)

## Reproduce the reported scores

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/Hubert0916/arsp.git
cd arsp
uv sync --locked
uv run hf auth login
uv run --locked python tools/reproduce.py
```

This command downloads the pinned public data, checks the split and frozen model
files, builds the actual submission, runs both predictors, and compares every
prediction with the published run (absolute tolerance `1e-7`). Acquired item IDs
and labels must match exactly. Outputs go to `results/reproduced/`.

Accept the access conditions on the
[measurement-db dataset page](https://huggingface.co/datasets/aims-foundations/measurement-db)
using your Hugging Face account before logging in. The upstream dataset is gated;
the download uses your own authorized read token, never a token stored in this
repository. The login step can be omitted when the pinned data is already cached.

The evaluation uses the **bundled frozen weights** without refitting. Inference
runs on CPU and calls no external APIs. Allow approximately 6 GB of disk for
the dataset tables, model files, prepared inputs, and outputs, plus the Python
environment. Download time depends on your connection. All 568 public Parquet
tables total approximately 4.62 GB; traces and raw archives are not downloaded.

If the pinned data is already in your Hugging Face cache:

```bash
uv run --locked python tools/reproduce.py --cache-only
```

To rerun inference using already prepared inputs (their SHA256 is still checked):

```bash
uv run --locked python tools/reproduce.py --prepared
```

## Evaluation protocol

- Public data: `aims-foundations/measurement-db`, revision
  `54074453da7b9911be0f48e4c560be547bc84d21`; 142 specified binary sources.
- Seed: **20261028**. SHA256 ordering selects one eligible subject per source and
  80 distinct items per subject. The first 40 form the acquisition stream; the
  remaining 40 form the evaluation set. Repeated item responses are resolved by
  the lowest `(trial, response_id)` among binary outcomes.
- Model inputs contain only the official visible subject/item fields, with nulls
  converted to empty strings. Benchmark IDs are anonymous SHA256 aliases.
  Separate grading/reference-answer columns, traces, and target outcomes are not
  passed to the predictor or acquisition hook. Item content is retained as
  provided upstream, including any examples already embedded in that content.
- The packaged acquisition hook is called sequentially. With 40 candidates and
  a cap of 31, warm-up is clipped to 9 skipped items, followed by 31 mandatory
  acquisitions. This setup **does not measure an active-selection gain**.
- Both ASRP and official Empirical Mean receive the same nested evidence at
  budgets `0, 1, 3, 7, 15, 31`, and score the same 40 targets per group.
- Brier scores are averaged **equally across groups**. Brier ALC uses weights
  `0.1, 0.2, 0.2, 0.2, 0.2, 0.1`, equivalent to normalized trapezoidal area on
  `log2(1+n)`.
- Frozen source guards are keyed by public source names. They do not match the
  anonymous benchmark aliases, so this experiment applies no source guard;
  topic calibration remains active.

These are local public-proxy results. Public training/calibration and proxy data
come from the same release; the online acquisition/evaluation split is not a
source-held-out training/test split. Official Codabench sampling and private data
can produce different scores.

<!-- RESULTS_START -->

| Method | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASRP | 0.2268 | 0.2245 | 0.2133 | 0.1981 | 0.1939 | 0.1852 | 0.2072 |
| Empirical Mean | 0.2500 | 0.3877 | 0.2505 | 0.2079 | 0.1987 | 0.1918 | 0.2531 |

<!-- RESULTS_END -->

The complete evidence is in [results/proxy](results/proxy): per-item predictions,
acquisition decisions and labels, per-source metrics, and run provenance. The
[split manifest](experiments/proxy_manifest.json) records data checksums and every
selected subject/item/response. Raw public task content stays in the local
`data/` directory and is fetched from upstream, respecting its existing licenses.

Recalculate metrics from published predictions without model inference:

```bash
uv run --locked python tools/score_proxy.py results/proxy
```

## Build competition submissions

```bash
uv run --locked python tools/verify_artifacts.py
uv run --locked python tools/build_submission.py asrp
uv run --locked python tools/smoke_test.py dist/asrp.zip
uv run --locked python tools/build_submission.py empirical_mean
uv run --locked python tools/smoke_test.py dist/empirical_mean.zip
```

Upload the appropriate ZIP in `dist/`. The builder merges `src/asrp/` and
`artifacts/asrp/` into a flat ZIP with `model.py` at its root. Run the packaged
version: inference loaders expect weights beside the source. ASRP includes
`labeling.py`; the standalone Empirical Mean ZIP uses the platform's default
random acquisition policy, whereas our matched-evidence comparison uses ASRP's
labels for both methods.

The smoke test uses synthetic inputs. For the official submission checker,
download the [competition starting kit](https://aims.stanford.edu/competition).
It is maintained upstream and is not vendored here.

## Rebuild the report

```bash
uv run --locked python tools/score_proxy.py results/proxy
uv run --locked python tools/update_report_results.py results/proxy
uv run --locked python report/make_proxy_plot.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
latexmk -cd -c -bibtex report/main.tex
```

The last two commands require LaTeX, latexmk, BibTeX, and TikZ. The cleanup keeps
`main.pdf`. Numeric table rows, result analysis, and the plot all derive from
`results/proxy/summary.json`; the method diagram is native TikZ.

## Repository layout

```text
src/asrp/                  Prediction and acquisition source code
artifacts/asrp/            Frozen parameters, settings, and MiniLM encoders
artifacts/manifest.sha256.json  Inference-file checksums
baselines/empirical_mean/  Official baseline predictor and MIT license
experiments/              Pinned protocol, split IDs, and public-data checksums
results/proxy/            Published per-item predictions, labels, and metrics
report/                   Paper source, PDF, figures, and generated table inputs
tools/                    Data preparation, evaluation, scoring, tests, packaging
.python-version + pyproject.toml + uv.lock  Locked CPU environment
data/                     Download-derived inputs (generated, not tracked)
dist/                     Submission ZIPs (generated, not tracked)
```

Both MiniLM encoders are bundled (about 173 MiB combined) as ordinary Git files,
not LFS pointers. Inference does not download models. See
[third-party notices](THIRD_PARTY_NOTICES.md). Codex assisted with development
and report preparation; it is not an online predictor.
