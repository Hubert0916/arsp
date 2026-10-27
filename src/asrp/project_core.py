"""Support-adaptive kernel regression with an exact Beta-mixture intercept.

Only revealed input features determine preprocessing and bandwidth. Outcomes
fit centered ridge residuals shared across subjects of the current benchmark.
Query features never change the fitted geometry, intercepts, or coefficients.
"""

import numpy as np
from scipy.special import betaln, logsumexp


def initial(offset, config):
    return np.clip(.5 + config["carry"] * np.asarray(offset), .05, .95)


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fit_geometry(semantic, lexical):
    semantic, lexical = np.asarray(semantic, float), np.asarray(lexical, float)
    mean = np.zeros(semantic.shape[1])
    semantic_weight = np.ones(semantic.shape[1])
    lexical_weight = np.ones(lexical.shape[1])
    if len(semantic) > 1:
        mean = semantic.mean(axis=0)
        variance = semantic.var(axis=0)
        semantic_weight = 1 / np.sqrt(variance + max(float(np.median(variance)), 1e-7))
        frequency = (np.abs(lexical) > 1e-10).sum(axis=0)
        lexical_weight = np.log((1 + len(lexical)) / (1 + frequency)) + .1
    geometry = mean, semantic_weight, lexical_weight
    return geometry, *transform(geometry, semantic, lexical)


def transform(geometry, semantic, lexical):
    mean, semantic_weight, lexical_weight = geometry
    return normalize((semantic - mean) * semantic_weight), normalize(lexical * lexical_weight)


def local_gram(values):
    _, semantic, lexical = fit_geometry(values[:, :384], values[:, 384:])
    return semantic @ semantic.T + lexical @ lexical.T


def mixture_mean(successes, counts, offsets, config):
    concentrations = np.asarray(config["concentrations"], float)
    masses = np.asarray(config["masses"], float)
    mean = initial(offsets, config)[:, None]
    alpha, beta = mean * concentrations, (1 - mean) * concentrations
    successes, counts = np.asarray(successes)[:, None], np.asarray(counts)[:, None]
    log_weights = (np.log(masses) + betaln(alpha + successes, beta + counts - successes)
                   - betaln(alpha, beta))
    weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    return np.sum(weights * (alpha + successes) / (concentrations + counts), axis=1)


def fit_evidence(semantic, lexical, labels, who, offsets, config):
    """Raw vectors in, reusable evidence state out (no query or future pool)."""
    geometry, semantic, lexical = fit_geometry(semantic, lexical)
    raw = (semantic @ semantic.T + lexical @ lexical.T) / 2
    distances = np.maximum(1 - raw[np.triu_indices(len(raw), k=1)], 0)
    positive = distances[distances > 1e-8]
    bandwidth = max(float(np.median(positive)) if len(positive) else .1, 1e-5) * config["kernel"][1]
    kernel = np.exp(np.minimum(raw - 1, 0) / bandwidth)
    who, labels = np.asarray(who, int), np.asarray(labels, float)
    counts = np.bincount(who, minlength=len(offsets))
    if np.any(counts == 0):
        raise ValueError("Every fitted subject must have at least one revealed label")
    successes = np.bincount(who, weights=labels, minlength=len(offsets))
    membership = np.eye(len(offsets))[who]
    average = (membership / counts).T
    center = np.eye(len(labels)) - membership @ average
    centered = center @ kernel @ center
    dual = center @ np.linalg.solve((centered + centered.T) / 2 + config["ridge"] * np.eye(len(labels)), center @ labels)
    return dict(geometry=geometry, semantic=semantic, lexical=lexical, bandwidth=bandwidth,
                base=mixture_mean(successes, counts, offsets, config), dual=dual,
                bias=average @ kernel @ dual)


def predict_fitted(state, semantic, lexical, qwho, config=None):
    semantic, lexical = transform(state["geometry"], semantic, lexical)
    raw = (semantic @ state["semantic"].T + lexical @ state["lexical"].T) / 2
    kernel = np.exp(np.minimum(raw - 1, 0) / state["bandwidth"])
    return np.clip(state["base"][qwho] + kernel @ state["dual"] - state["bias"][qwho], 0, 1)


def predict(fit, config, offsets, rate=None):
    """Offline adapter; input Gram matrices have already had local preprocessing."""
    raw = (fit.semantic + fit.lexical) / 2
    distances = np.maximum(1 - raw[np.triu_indices(len(raw), k=1)], 0)
    positive = distances[distances > 1e-8]
    bandwidth = max(float(np.median(positive)) if len(positive) else .1, 1e-5) * config["kernel"][1]
    kernel = np.exp(np.minimum(raw - 1, 0) / bandwidth)
    cross = np.exp(np.minimum((fit.semantic_cross + fit.lexical_cross) / 2 - 1, 0) / bandwidth)
    centered = fit.h @ kernel @ fit.h
    dual = fit.h @ np.linalg.solve((centered + centered.T) / 2 + config["ridge"] * np.eye(len(raw)), fit.residual)
    base = mixture_mean(fit.s, fit.n, offsets, config)
    return np.clip(base[fit.qwho] + (cross - fit.center @ kernel) @ dual, 0, 1)
