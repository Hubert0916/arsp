# ASRP technical report

`main.tex` uses the supplied NeurIPS 2026 template. The main text has seven sections;
mathematical details, source inventory, official submission IDs, reproduction, and
sampling assumptions are in the appendices. Keep the author/coauthor fields as
intended before submitting.

Current local results use the [public formative simulation](../results/formative):
30 complete-group draws, fixed within-group pools, all evaluation responses,
Random/Adaptive acquisition. Component-removal results remain available in the
repository; the report summarizes their limited evidence in the limitations. The only official
scores reported are the ten with-Rasch evaluations in `../results/official`.

From the repository root, audit, generate figures, compile, and clean intermediates:

```bash
uv run --locked python tools/score_draws.py results/formative
uv run --locked python tools/update_formative_report.py
uv run --locked python report/make_proxy_plot.py
uv run --locked python tools/update_official_report.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
latexmk -cd -c -bibtex report/main.tex
```

For a complete inference rerun, follow `../README.md` and run `tools/reproduce.py`.
Frozen-artifact reproduction does not retrain the original priors or calibration.
Older result generators are for historical protocols and should not be used to
update the current report.

The method figure is native TikZ in `figures/asrp_method.tex`; the Brier curve is
built by `make_proxy_plot.py`. Both retain the requested maximum width of
`0.95\columnwidth`, maximum height of `0.24\textheight`, and aspect ratio.
Table captions remain below their tables. The NeurIPS checklist is omitted.
