# Official Codabench evaluations

The report presents two separate batches, both using adaptive acquisition:

| Method | Completed runs | Mean Brier ALC | Sample SD |
|---|---:|---:|---:|
| With Rasch (retained submission) | 10 | 0.19107 | 0.01105 |
| Without Rasch | 16 | 0.20893 | 0.02813 |

These batches use different random draws and dates. They are **not a paired
Rasch ablation**. The controlled public ablations are in [`../ablation`](../ablation)
and [`../rasch_confirmation`](../rasch_confirmation). The official comparison
motivates caution about removing Rasch but does not establish its causal effect.

- `runs.json`, `summary.json`: ten with-Rasch evaluations, with all six budgets;
  equal groups within each run, equal runs across the batch.
- `submissions.json`: their platform IDs matched to rounded feedback ALC.
- `without_rasch_screenshot.json`: all 17 rows from the later screenshot; all
  16 Finished entries are included and Cancelled submission 1020351 excluded.
  Only total ALC is supplied for this batch; no budget scores are inferred.
- `comparison.json`: generated means, SDs, and included/excluded IDs.
- `status.json`: restored local inference version and manifest checksum.
- `provenance.json`, `ingestion.json`: origins and checks for the earlier feedback.

The author confirmed one ZIP across the earlier ten evaluations. The current
local inference files match the pre-removal Git revision. Historical platform
ZIP hashes are unavailable. The later batch's no-Rasch association follows the
submission discussion and screenshot; filenames alone cannot establish contents.
Original feedback is retained locally and excluded from the public repository.

To verify aggregates and regenerate the report tables:

```bash
uv run --locked python tools/update_official_report.py
```

Private platform evaluations cannot be reproduced from public data. The local
parser `tools/parse_official_results.py` can audit detailed platform feedback;
its raw transcription is not published.
