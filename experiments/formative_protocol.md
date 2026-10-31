# Public simulation of formative evaluation

This protocol replaces the report's fixed-subject, three-split proxy. It follows
the public PAIEC evaluation rules on a pinned **public** dataset snapshot. It is
not an exact replay of the private Codabench sampler or a source-held-out test.
The predictor and predictive weights are retained; the current submission uses uniform random acquisition.

| Protocol feature | Implementation |
|---|---|
| Complete groups | Every binary subject–benchmark group with 80–1,000 distinct items is eligible; no item truncation. |
| Per-draw cap | At most 1,000 distinct subject–item pairs across acquisition and evaluation. |
| Repeated draws | 30 seeds fixed in `formative_config.json`; subjects can change across draws. |
| Balanced selection | Among groups that fit, choose a least-represented benchmark, then a least-represented subject within it; randomize ties. |
| Persistent pools | One snapshot seed fixes a 50/50 item split and candidate order; all response runs of an item stay together. |
| Acquisition | Uniform streaming acquisition with probability L/R; one recorded response per acquired item; nested prefixes at 0, 1, 3, 7, 15, 31. |
| Evidence | All groups complete a checkpoint before evaluation; predictors receive the complete acquired evidence across groups. |
| Repeated outcomes | Score every binary response in the evaluation pool; identical visible inputs reuse their prediction at a checkpoint. |
| Scoring | Average response losses within group, groups within draw, then draws equally; preserve the official ALC weights. |
| Comparisons | ASRP and Empirical Mean share draws, pools, candidate order, and randomly acquired labels. Diagnostic prediction ablations use the same evidence. |

## Explicit local choices

The public starting kit exposes the streaming client, not the private coordinator.
It describes balanced group selection but does not specify the full sampler or
its RNG. The local implementation uses hierarchical benchmark-first tie-breaking,
SHA256 ranks, stable public-ID aliases, a fixed hash-selected recorded acquisition
response, and sequential group processing within each checkpoint. These choices
are recorded rather than claimed to be organizer code. Private pool contents,
pool assignments, aliases and random seeds cannot be reproduced from public data.

Some public subjects have identical visible dictionaries because metadata are
missing. Their upstream identities are retained, and the submitted predictor's
native behavior is used rather than inventing additional visible attributes.

## Sources and provenance

- Official rules: <https://aims.stanford.edu/competition>, accessed 2026-10-31.
- Local official starting-kit `README.md`: SHA256 recorded in the configuration;
  its “Streaming test-time adaptation” and draw-limit description specify fixed
  pools, complete-group sampling, all evaluation responses, and global evidence.
- Starting-kit `tools/streaming_ingestion.py`: SHA256 recorded in the configuration.
  The default random-acquisition formula matches that public client.
- Public data, file checksums, eligible inventory, selected groups, all response
  IDs and draw membership are recorded in `formative_manifest.json`.

The preliminary group-level tie implementation was corrected to benchmark-first
selection before score inspection. The released configuration and results use
only the final sampler, with no selection based on measured performance.
