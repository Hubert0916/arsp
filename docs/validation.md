# Local release validation

Checked on 2026-10-28 in the CPU environment recorded by `uv.lock`.

| Check | Result |
|---|---|
| `uv sync --locked` / environment setup | Passed |
| Frozen inference artifact hashes | All 38 files matched before and after directory reorganization |
| ASRP submission ZIP checker | Passed |
| ASRP official synthetic smoke test | Passed (2 predictions) |
| Empirical Mean submission ZIP checker | Passed |
| Empirical Mean official synthetic smoke test | Passed (2 predictions) |
| ASRP prediction sanity checks at 0, 1, 3, 7, 15, 31 labels | Finite probabilities in [0, 1] |
| Synthetic streaming acquisition, 80 candidates | First 16 skipped; exactly 31 labels acquired |
| Empirical Mean reconstruction from retained public rows | ALC 0.242221711 |
| Report LaTeX compilation | Passed |
| Selected text-file scan for credential patterns | No matches |
| GitHub single-file size limit | All files below 100 MiB |

The synthetic checks validate packaging and runtime behavior, not predictive
performance. They do not reproduce the final ASRP public-proxy scores or provide
new hidden-test results. See the reproducibility scope in ../README.md.

## Directory reorganization

Source and artifacts are packaged together from `src/asrp/` and `artifacts/asrp/`.
The source and inference artifact bytes were preserved. The official starting kit
is retained outside this repository for local validation only. The repository
provides its own synthetic six-budget smoke test, which is not the official evaluator.
