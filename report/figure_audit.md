# ASRP method figure audit

This file records the text-only audit requested by `paper-framework-figure-studio-pro` before rendering the method overview. The final figure is an editable TikZ line-art schematic in `figures/asrp_method.tex`; no generated image was used because the local skill documents that its current v3.2.15f image stage is designed for ChatGPT Web and Codex cannot supply the required reference-image input.

## S0: paper foundation

**Reader goal.** Understand what is trained offline, what is computed for one subject--item query, how acquired labels change the prediction, and where the final probability is emitted.

**Nodes.** Public measurement-db; MiniLM and metadata features; frozen difficulty and subject-prior artifacts; topic residuals and eight fallback centroids; subject--item input; item encoding; Rasch hybrid; Kernel Probe; budget-specific ensemble with source guard; topic calibration; final probability; candidate stream; `adaptive_4_warm`; revealed labels.

**Edges.** Offline data -> features -> frozen artifacts. Input -> encoding -> Rasch and Kernel Probe. Both predictors -> budget ensemble with source guard -> topic calibration -> final probability. Candidate stream -> acquisition -> revealed labels -> both adaptive predictors. Frozen artifacts feed their corresponding runtime modules.

**Evidence anchors.** `project/asrp/model.py`, `runtime_config.json`, `labeling.py`, `project_policy.py`, and `project/asrp/README.md`. The six budgets and the `adaptive_4_warm` hook are specified in the starting kit and the submission code.

**Risk controls.** No online API, hidden labels, or benchmark lookup is shown. Topic calibration is marked as a post-ensemble correction, and the source guard is marked as a weight adjustment rather than an oracle. The warm-up label is stated as “up to 16” because the implementation clips it when the remaining candidate count must support the 31-label cap.

## S1: figure strategy

The figure type is a two-lane method overview with a compact feedback loop. The reader path is left-to-right: offline artifacts, runtime prediction, then revealed-label feedback. Offline nodes use a pale blue fill, online nodes use a pale orange fill, the final output uses pale green, and dashed arrows denote information available across the offline/online boundary or feedback loop. The figure uses an ACM/IEEE/AAAI-style line-art schematic: white background, thin strokes, no gradients, no decorative icons, and no long prose inside nodes.

## S4: visible text contract

The only visible mathematical variables are $[s,i]$, $p_{\rm hybrid}$, $p_{\rm kernel}$, $p(\mathrm{correct})$, and the acquired-label set. The caption and Sections 4--5 carry the parameter values and algorithm details. Text inside the figure is restricted to module names, the public-data boundary, the acquisition policy name, and the six budgets.

## Render/negative constraints

The figure must remain legible when reduced to a NeurIPS two-column width, keep all arrows connected to their intended nodes, and avoid implying that evaluation outcomes enter acquisition. It must not show an LLM API, a hidden-test data path, a per-source oracle, or an unbounded warm-up phase.
