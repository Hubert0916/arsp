# ASRP technical report

`main.tex` follows the supplied NeurIPS 2026 template and includes the method figure, public-proxy table, appendix source inventory, and bibliography. The NeurIPS Paper Checklist is omitted because it is not required by the AIMS competition rules.

The report presents only local public-proxy results for ASRP and Empirical Mean. The numbers in `results_proxy.json` are not hidden-test results. The experimental setup explains that random formative draws and public/private distribution differences can make Codabench scores better or worse than these local values.

Build locally with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The method figure is editable TikZ in `figures/asrp_method.tex`. The pre-generation semantic/visual/text audit required by `paper-framework-figure-studio-pro` is recorded in `figure_audit.md`.

The public release is https://github.com/Hubert0916/arsp. Its README documents the available frozen inference artifacts and the missing original training/final-evaluation scripts. Historical diagnostics are not the final ASRP prediction records.
