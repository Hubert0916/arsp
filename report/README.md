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
uv run --locked python tools/update_ablation_report.py
uv run --locked python tools/update_rasch_report.py
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

The main text has seven sections; reproduction and disclosure are in Appendix D.
Section 3 presents the core prediction and acquisition equations. Appendix A
uses the same notation and follows Sections 3.1–3.5 in order: feature construction,
prior construction, kernel/Beta calculations, mixing/calibration, and acquisition.
Kernel mixing weights (w), ridge penalty (lambda_K), and acquisition gain/utility
(G/V) have separate symbols and explicit code mappings.

Component ablations hold the published Adaptive evidence fixed across all 92
groups and three seeds, removing one component at a time without refitting.
Component predictions and native-runtime checks are produced by
`tools/evaluate_ablation.py`; see [`results/ablation`](../results/ablation).
The Rasch comparison uses six paired seeds; see
[`results/rasch_confirmation`](../results/rasch_confirmation). Its evidence
supports a small zero-label benefit, not an overall local ALC advantage.

## Official evaluation

The report restores the ten detailed with-Rasch evaluations. A separate unpaired
comparison uses all 16 completed no-Rasch entries in the later platform
screenshot, excluding its cancelled entry. No missing budget scores are inferred.
Appendix C lists both batches' submission IDs.

`tools/update_official_report.py` regenerates the tables from published records,
checks the old feedback summary, and verifies the restored inference manifest.
Historical platform ZIP hashes are unavailable; method association is documented
in [`results/official`](../results/official).

The public reproduction pipeline uses frozen artifacts. It does not retrain the
original priors or calibration; see
[`artifacts/README.md`](../artifacts/README.md). Keep the generated numeric inputs
in sync by running the commands above after changing the scored results.
