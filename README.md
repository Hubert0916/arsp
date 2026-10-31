# ASRP: Adaptive Support-Based Response Prediction

**Yuhan Chiu — Data Science Degree Program, National Taiwan University and Academia Sinica**

ASRP predicts success with 0–31 revealed labels using a Rasch hybrid, Kernel Probe,
topic calibration, and **uniform random acquisition**. Prediction adapts to the revealed support; label selection uses the streaming probability `labels_remaining / items_remaining`.

[Report](report/main.pdf) · [Method code](src/asrp) · [Report source](report/README.md)

## Evaluation

The current public simulation uses **30 complete-group draws capped at 1,000
subject–item pairs each**. Subjects vary between draws; within-group pools remain
fixed, and every evaluation response is scored. Both predictors share the same randomly acquired labels. See the [protocol](experiments/formative_protocol.md)
and [released predictions and ablations](results/formative).

<!-- RESULTS_START -->

| Predictor (Random acquisition) | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empirical Mean | 0.2500 | 0.3439 | 0.2161 | 0.1909 | 0.1794 | 0.1745 | 0.2285 |
| ASRP | 0.2111 | 0.2032 | 0.1924 | 0.1830 | 0.1778 | 0.1707 | 0.1895 |

<!-- RESULTS_END -->

This follows the documented formative design on public data. The private sampler
and test pool are unavailable; local choices are explicit. Frozen fitting and
calibration overlap the public release, so these are not source-held-out results.

**Official Codabench:** results for this Random-acquisition release are pending.
Earlier evaluations used a different acquisition policy and are retained only as
[historical records](results/official/historical_adaptive).

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
and `results/rasch_*`; they are not mixed into the current report. To reproduce historical paired-policy experiments, use Git revision `ee9257a`; the current runtime supports Random acquisition only.

## Submission and files

```bash
uv run --locked python tools/verify_artifacts.py
uv run --locked python tools/build_submission.py asrp
uv run --locked python tools/smoke_test.py dist/asrp.zip
```

Upload `dist/asrp.zip` to Codabench. Replace `asrp` with `empirical_mean` to package
the official baseline. This ZIP replaces the earlier acquisition policy with Random; a new Codabench submission is required to evaluate it.

- `src/asrp`, `artifacts`: inference code and frozen models.
- `experiments`, `tools`: protocols, manifests, and reproduction scripts.
- `results`, `report`: predictions, scores, paper, and figures.

Keep `data/`, the Hugging Face cache, `artifacts/`, and `experiments/` for local
reruns. `proxy_manifest.json`, `evaluate_proxy.py`, and `evaluate_ablation.py`
remain dependencies of the current pipeline despite their historical names.
Temporary result `chunks/` are regenerated during inference; the released
compressed predictions and acquisition traces are the verification reference.

See [third-party notices](THIRD_PARTY_NOTICES.md). Codex assisted development and
writing; no external LLM API is used during scoring.
