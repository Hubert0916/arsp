# ASRP submission

ASRP (Adaptive Support-Based Response Prediction) combines:

- frozen public success counts for the subject and visible item metadata;
- evidence smoothing of that query-specific prior;
- Kernel Probe residuals fitted to revealed labels and item similarities;
- shrunk, capped topic corrections, with frozen MiniLM centroids as fallback.

The kernel receives weights 0, 0, 0.2, 0.4, 1, and 1 at label budgets
0, 1, 3, 7, 15, and 31. `runtime_config.json` stores these weights and the
smoothing concentration. `labeling.py` uses `adaptive_4_warm` acquisition.
`text_encoder.py` supplies embeddings for topic-cluster assignment; kernel and
acquisition features use the packaged ONNX encoder.

The predictor reads only the supplied subject/item attributes and revealed
binary outcomes. Frozen artifacts come from public measurement-db responses.
The current version has no Rasch difficulty network or online ability update.
Earlier-version comparisons are retained under `results/legacy_rasch` and
`results/rasch_confirmation` in the repository.
