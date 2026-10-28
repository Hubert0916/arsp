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
files, builds the actual submission, runs both predictors with both acquisition policies at seeds 42, 43, and 44, and compares every
prediction with the published run (absolute tolerance `1e-7`). Acquired item IDs
and labels must match exactly. Outputs go to `results/reproduced/`.

Accept the access conditions on the
[measurement-db dataset page](https://huggingface.co/datasets/aims-foundations/measurement-db)
using your Hugging Face account before logging in. The upstream dataset is gated;
the download uses your own authorized read token, never a token stored in this
repository. The login step can be omitted when the pinned data is already cached.

The evaluation uses the **bundled frozen weights** without refitting. Inference
runs on CPU and calls no external APIs. Four independent worker processes are used by default;
pass `--workers 1` on a smaller machine. Allow approximately 6 GB of disk for
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
  `54074453da7b9911be0f48e4c560be547bc84d21`; a fixed inventory of 142 binary sources.
- Eligibility restores **complete 80–1,000-item groups**. Groups outside this range
  are excluded, not truncated. There are 92 sources with an eligible group and
  50 without one; the manifest records all inclusion/exclusion decisions.
- One eligible subject per included source is selected by SHA256 ordering with
  group-selection seed **20261028**, fixed across all comparisons. The 92 selected
  groups contain **44,741 items**, ranging from 90 to 1,000 per group. This is a
  public-proxy sample, not an evaluation of every eligible subject.
- Split/acquisition seeds are **42, 43, 44**. For each seed, SHA256 item ordering
  assigns `ceil(N/2)` items to acquisition and `floor(N/2)` to evaluation. Candidate
  pools range from **45 to 500** items; every seed has **22,357 evaluation items**.
  Repeated responses are resolved by the lowest `(trial, response_id)` among
  binary outcomes, held fixed across policies and seeds.
- **Four combinations:** ASRP + Random, ASRP + Adaptive, Empirical Mean + Random,
  and Empirical Mean + Adaptive. Both policies use the same candidate order and
  evaluation pool within each seed. Both predictors receive exactly the same
  acquired labels within each policy.
- Random acquisition uses uniform sampling without replacement, accepting each
  candidate with probability `labels_remaining / items_remaining`. A hash of
  current input/context supplies deterministic random variates; anonymous subject
  aliases include the seed. This matches the sampling distribution, not the
  private server's exact random draws.
- Adaptive acquisition calls the unchanged packaged `adaptive_4_warm` hook. It
  skips `min(16, A-31)` warm-up candidates for a pool of size `A`, then uses
  covariance gain when choices remain. It must acquire all remaining candidates
  if their number equals the remaining label budget. Small eligible groups can
  still have no discretionary selections.
- A prespecified **at-least-160-item subset** contains 78 groups, each with at
  least 80 acquisition candidates. Results for this subset show policy effects
  when there is more room for selection after warm-up.
- Six budgets: `0, 1, 3, 7, 15, 31`. Brier scores are averaged within groups,
  **equally across groups**, then equally across the three seeds. Pooling all
  prediction rows is incorrect because group sizes vary. Brier ALC uses weights
  `0.1, 0.2, 0.2, 0.2, 0.2, 0.1` on the `log2(1+n)` axis.
- Model inputs contain only official visible subject/item fields and anonymous
  benchmark aliases. Separate grading/reference-answer columns, traces, and
  target outcomes are not passed to the predictor/hook. Upstream item content,
  including any embedded examples, is retained as supplied.
- Topic calibration is active. Frozen model weights are identical across all runs.

These are local public-proxy results. Public training/calibration and proxy data
come from the same release; the online acquisition/evaluation split is not a
source-held-out training/test split. Three seed scores and their sample standard
deviation summarize these draws, not a confidence interval over all evaluations.
Official Codabench sampling and private data can produce different scores.

<!-- RESULTS_START -->

| Predictor + acquisition | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empirical Mean + Random | 0.2500 | 0.3488 | 0.2518 | 0.2086 | 0.1910 | 0.1842 | 0.2435 |
| Empirical Mean + Adaptive | 0.2500 | 0.3594 | 0.2366 | 0.2003 | 0.1883 | 0.1837 | 0.2403 |
| ASRP + Random | 0.2263 | 0.2166 | 0.2075 | 0.1947 | 0.1845 | 0.1755 | 0.2008 |
| ASRP + Adaptive | 0.2263 | 0.2161 | 0.2043 | 0.1912 | 0.1819 | 0.1738 | 0.1987 |

<!-- RESULTS_END -->

The complete evidence is in [results/proxy](results/proxy): per-item predictions,
acquisition decisions with warm-up/discretionary/forced reasons, labels,
per-group/seed/policy metrics, acquisition effects, and run provenance. The
[split manifest](experiments/proxy_manifest.json) records data checksums and every
selected subject/item/response. Raw public task content stays in the local
`data/` directory and is fetched from upstream, respecting its existing licenses.

Recalculate metrics from published predictions without model inference:

```bash
uv run --locked python tools/score_proxy.py results/proxy
```

## Official formative feedback

Ten complete, distinct feedback records supplied for ASRP with adaptive
acquisition have **mean Brier ALC 0.19107 (sample SD 0.01105)**. The submitter
confirmed that every evaluation used the same ASRP + `adaptive_4_warm` ZIP.
These private-platform formative samples are separate from the matched
public-proxy comparison above; no paired official baseline feedback was supplied.

The final attachment contains ten complete, distinct feedback blocks and replaces
the earlier incomplete copies. All six budgets are available; all metrics use
the same ten records.
See [aggregate results](results/official/summary.json),
[per-record aggregates](results/official/runs.json), and
[provenance](results/official/provenance.json). The original copy and restored
feedback remain local in `results/official/raw/`.

```bash
uv run --locked python tools/update_official_report.py
```

This recomputes the summary from released aggregates and regenerates report inputs;
it does not rerun private platform evaluation.

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
random acquisition policy, whereas our paired experiment runs both predictors under each policy.

The smoke test uses synthetic inputs. For the official submission checker,
download the [competition starting kit](https://aims.stanford.edu/competition).
It is maintained upstream and is not vendored here.

## Rebuild the report

```bash
uv run --locked python tools/score_proxy.py results/proxy
uv run --locked python tools/update_report_results.py results/proxy
uv run --locked python report/make_proxy_plot.py
uv run --locked python tools/update_official_report.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
latexmk -cd -c -bibtex report/main.tex
```

The last two commands require LaTeX, latexmk, BibTeX, and TikZ. The cleanup keeps
`main.pdf`. Numeric table rows, dataset counts, source inventory, result analysis, and the plot all derive from
`results/proxy/summary.json`; the method diagram is native TikZ.

## Repository layout

```text
src/asrp/                  Prediction and acquisition source code
artifacts/asrp/            Frozen parameters, settings, and MiniLM encoders
artifacts/manifest.sha256.json  Inference-file checksums
baselines/empirical_mean/  Official baseline predictor and MIT license
experiments/              Pinned protocol, split IDs, and public-data checksums
results/proxy/            Published per-item predictions, labels, and metrics
results/official/         Official feedback aggregates and ingestion audit
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
