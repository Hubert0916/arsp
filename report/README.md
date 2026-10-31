# ASRP technical report

`main.tex` uses the NeurIPS 2026 template and the seven-section structure. Author/coauthor fields are retained for completion before submission.

The report compares ASRP and Empirical Mean on 30 complete-group public draws with identical randomly acquired labels. It presents only local public-data results. Section 5.3 reports matched-evidence removals of Rasch, the kernel, and topic calibration, including all six budget scores and ALC differences. These tables and their analysis are generated from `results/formative/summary.json` by `tools/update_formative_report.py`.

From the repository root:

```bash
uv run --locked python tools/score_draws.py results/formative
uv run --locked python tools/update_formative_report.py
uv run --locked python report/make_proxy_plot.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/main.tex
latexmk -cd -c -bibtex report/main.tex
```

For an inference rerun, use `tools/reproduce.py` as described in the repository README. Reproduction uses frozen artifacts, without retraining the original priors or calibration.

The method figure is TikZ in `figures/asrp_method.tex`. Figure 2 is a single line chart of mean Random-acquisition Brier scores, with serif fonts, thin lines, and distinct filled-circle/open-square markers. Both figures retain width `0.95\columnwidth`, height `0.24\textheight`, and `keepaspectratio`. Table captions remain below their tables. The NeurIPS checklist is omitted.
