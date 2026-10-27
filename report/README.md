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
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
latexmk -cd -c -bibtex report/main.tex
```

The score table, per-seed table, dataset-count macros, source inventory, analysis,
and conclusion inputs are generated from the scored run. Complete 80–1,000-item
groups are split in half; the comparison uses seeds 42, 43, and 44.
Edit methodological text in `main.tex` and the native TikZ method figure in
`figures/asrp_method.tex`. Figure 2 retains the requested maximum width of
`0.95\columnwidth` and maximum height of `0.24\textheight`, preserving aspect ratio.

Per-item predictions and acquisition trajectories are released in
[`results/proxy`](../results/proxy). These are public-proxy results, not private
Codabench results. The NeurIPS checklist is not included in this competition report.
