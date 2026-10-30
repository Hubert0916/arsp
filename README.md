# ASRP: Adaptive Support-Based Response Prediction

**Yuhan Chiu — National Taiwan University**

ASRP predicts success with 0–31 revealed labels using a Rasch hybrid, Kernel Probe,
topic calibration, and adaptive acquisition.

[Report](report/main.pdf) · [Method code](src/asrp) · [Report source](report/README.md)

## Evaluation

The current public simulation uses **30 complete-group draws capped at 1,000
subject–item pairs each**. Subjects vary between draws; within-group pools remain
fixed, and every evaluation response is scored. Predictors share labels within
Random or Adaptive acquisition. See the [protocol](experiments/formative_protocol.md)
and [released predictions and ablations](results/formative).

<!-- RESULTS_START -->

| Predictor + acquisition | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empirical Mean + Random | 0.2500 | 0.3439 | 0.2161 | 0.1909 | 0.1794 | 0.1745 | 0.2285 |
| Empirical Mean + Adaptive | 0.2500 | 0.3462 | 0.2139 | 0.1931 | 0.1805 | 0.1765 | 0.2294 |
| ASRP + Random | 0.2111 | 0.2032 | 0.1924 | 0.1830 | 0.1778 | 0.1707 | 0.1895 |
| ASRP + Adaptive | 0.2111 | 0.2061 | 0.1915 | 0.1847 | 0.1797 | 0.1720 | 0.1907 |

<!-- RESULTS_END -->

This follows the documented formative design on public data. The private sampler
and test pool are unavailable; local choices are explicit. Frozen fitting and
calibration overlap the public release, so these are not source-held-out results.

**Official Codabench:** ten with-Rasch ASRP evaluations have mean Brier ALC
**0.19107** (sample SD **0.01105**). [Aggregates](results/official).

## Reproduce

Install [uv](https://docs.astral.sh/uv/) and accept the access conditions for
[measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db):

```bash
git clone https://github.com/Hubert0916/arsp.git
cd arsp
uv sync --locked
uv run hf auth login
uv run --locked python tools/reproduce.py
```

This reconstructs the pinned public data and draws, runs frozen inference on CPU,
and compares every prediction and acquisition decision with the release. Outputs
are in `results/check-formative/`. Use `--cache-only` for cached data, `--prepared`
for verified prepared data, or `--workers 1` on a smaller machine. The public tables
total about 4.6 GB, plus prepared data, outputs, models and the Python environment.
Original offline fitting scripts are not included.

Audit scores without model inference:

```bash
uv run --locked python tools/score_draws.py results/formative
uv run --locked python tools/update_formative_report.py
```

Earlier fixed-subject experiments remain under `results/proxy`, `results/ablation`,
and `results/rasch_*`; they are not mixed into the current report. Use
`tools/reproduce.py --legacy` to reproduce the old main proxy.

## Submission and files

```bash
uv run --locked python tools/verify_artifacts.py
uv run --locked python tools/build_submission.py asrp
uv run --locked python tools/smoke_test.py dist/asrp.zip
```

Upload `dist/asrp.zip` to Codabench. Replace `asrp` with `empirical_mean` to package
the official baseline. The evaluation revision does not alter the submitted model.

- `src/asrp`, `artifacts`: inference code and frozen models.
- `experiments`, `tools`: protocols, manifests, and reproduction scripts.
- `results`, `report`: predictions, scores, paper, and figures.

See [third-party notices](THIRD_PARTY_NOTICES.md). Codex assisted development and
writing; no external LLM API is used during scoring.
