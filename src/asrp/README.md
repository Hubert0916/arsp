# ASRP submission

ASRP (Adaptive Support-Based Response Prediction) combines the following components:

- a public-training prior keyed by the runtime `normalized_name` and visible `item_features`;
- a Rasch-style item difficulty estimate from frozen MiniLM features;
- the support-conditioned kernel Probe after labels are acquired.
- a conservative public-data calibration layer with topic residual shrinkage
  and frozen MiniLM centroids for items without
  topic metadata.

The metadata prior uses only public `measurement-db` training responses. At test time it reads the query's `item_features`; the current conservative configuration updates this prior with the aggregate revealed labels and lets the kernel Probe use the acquired item embeddings. No benchmark response lookup or private item data is used.

`runtime_config.json` contains the frozen budget-specific blend. `labeling.py` uses the `adaptive_4_warm` acquisition policy. The six budgets are 0, 1, 3, 7, 15, and 31 labels.

`topic_calibration.json` is generated only from public binary responses. Topic
corrections are shrunk toward zero and capped. If this artifact is absent, the predictor
falls back to the original ASRP path.
