# ASRP technical report

`main.tex` follows the supplied NeurIPS 2026 template. The report compares ASRP
with official Empirical Mean under Random and Adaptive acquisition on three
paired public-data splits described in
`../experiments/proxy_config.json`.

From the repository root, reproduce the scores with:

```bash
uv sync --locked
uv run hf auth login
uv run --locked python tools/reproduce.py
```

Accept the upstream measurement-db access conditions before logging in. Cached
users can instead run `tools/reproduce.py --cache-only` without a network download.

Generate numeric report inputs and compile:

```bash
uv run --locked python tools/score_proxy.py results/proxy
uv run --locked python tools/update_report_results.py results/proxy
uv run --locked python report/make_proxy_plot.py
uv run --locked python tools/update_official_report.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
latexmk -cd -c -bibtex report/main.tex
```

The primary score table, per-seed table,
dataset-count macros, source inventory, and analysis inputs are generated from the scored run. Complete 80–1,000-item
groups are split in half; the comparison uses seeds 42, 43, and 44.
Edit methodological text in `main.tex` and the native TikZ method figure in
`figures/asrp_method.tex`. Figure 2 retains the requested maximum width of
`0.95\columnwidth` and maximum height of `0.24\textheight`, preserving aspect ratio.

Per-item predictions and acquisition trajectories are released in
[`results/proxy`](../results/proxy). These are public-proxy results, not private
Codabench results. The NeurIPS checklist is not included in this competition report.

Section 4 presents the core prediction and acquisition equations. Appendix A
uses the same notation and follows Sections 4.1–4.5 in order: feature construction,
priors and MAP updates, kernel/Beta calculations, mixing/calibration, and acquisition.
Rasch mixing weights (eta), kernel ridge penalty (lambda_K), and acquisition
gain/utility (G/V) have separate symbols and explicit code mappings.

## Official formative feedback

The report separately summarizes ten complete formative evaluations of the same
ASRP + `adaptive_4_warm` ZIP, identified by Codabench submission IDs in Appendix C.

`../results/official/runs.json` contains evaluation-level aggregates, and
`summary.json` contains equal-record means and sample standard deviations.
The report generator recomputes and verifies these statistics before generating
its tables. It needs no private feedback file.

The original feedback is retained locally at
`../results/official/raw/official_result.log` and is not tracked. When that private
file is available, re-import it with:

```bash
uv run --locked python tools/parse_official_results.py
uv run --locked python tools/update_official_report.py
```

The parser checks group joins, counts, all six budgets, ALC and ECE consistency,
and duplicate feedback. Anonymous platform group IDs are not exported. These
commands reproduce aggregation, not the platform's private evaluation draws.

The public reproduction pipeline uses frozen artifacts. It does not retrain the
original priors, difficulty network, or calibration; see
[`artifacts/README.md`](../artifacts/README.md). Keep the generated numeric inputs
in sync by running the commands above after changing the scored results.
