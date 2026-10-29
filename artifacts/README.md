# Frozen inference artifacts

The released evaluation uses these files directly. Verify their checksums with
`uv run --locked python tools/verify_artifacts.py` from the repository root.

| File in `asrp/` | Runtime role |
|---|---|
| `encoder/` | Frozen MiniLM encoder for topic-cluster assignment. |
| `probe_encoder/` | Frozen ONNX MiniLM encoder for kernel and acquisition features. |
| `binary_subject_prior.json` | Model-name offsets for the kernel Beta prior mean. |
| `feature_prior.json` | Public subject/metadata success counts for the query-specific smoothed prior. |
| `topic_calibration.json` | Budget-specific topic residuals, counts, eight centroids, and correction caps. |
| `runtime_config.json` | Smoothing strength and budget-specific kernel weights. |
| `probe_config.json` | Beta-mixture, kernel, ridge, and acquisition configuration. |

The final kernel weights are `probe_weights` in `asrp/runtime_config.json`.
Appendix A of the [report](../report/main.pdf) gives the equations and code
mapping. The build tool copies `asrp/` and `src/asrp/` into a flat submission ZIP.

Competition-specific artifacts use public `aims-foundations/measurement-db`
responses; the pretrained encoders are `sentence-transformers/all-MiniLM-L6-v2`.
See [third-party notices](../THIRD_PARTY_NOTICES.md).

`tools/reproduce.py` reproduces predictions and scores with these frozen files.
The original offline fitting scripts and artifact-specific training-row manifests
are not included. `experiments/proxy_manifest.json` describes the evaluation
selection, not the original training selection. Historical `research_files`
entries inside configuration files are provenance hashes, not runnable paths in
this release.
