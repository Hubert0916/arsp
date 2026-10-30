"""Experimental ASRP predictor with evidence-adaptive priors and no Rasch model."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from base_features import subject_key
from kernel_probe import get_predictor
from prior_features import _feature_prior, _calibration_adjust
from prior_kernel import probabilities
from project_core import transform
from text_encoder import MiniLMEncoder, normalize_text

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _resources():
    return {
        name: json.loads((HERE / filename).read_text())
        for name, filename in [
            ('feature', 'feature_prior.json'),
            ('topic', 'topic_calibration.json'),
            ('config', 'runtime_config.json'),
        ]
    }


@lru_cache(maxsize=1)
def _encoder():
    return MiniLMEncoder(HERE / 'encoder')


@lru_cache(maxsize=4096)
def _embedding(text):
    return _encoder().encode([normalize_text(text)])[0]


@lru_cache(maxsize=4096)
def _prior(serialized_subject, metadata):
    return _feature_prior(json.loads(serialized_subject), {'item_features': metadata}, _resources()['feature'])


def _query_prior(x):
    return _prior(json.dumps(x[0], sort_keys=True), x[1].get('item_features', ''))


def prediction_parts(input, labeled=None):
    """Use only the query, frozen artifacts, and evaluator-revealed evidence."""
    evidence = labeled or []
    subject, item = input
    own = []
    for x, y in evidence:
        if x[0] == subject and x[1].get('benchmark_id') == item.get('benchmark_id'):
            if y not in (0, 1):
                raise ValueError('Acquired outcomes must be binary.')
            own.append((x, y))
    own = own[:31]
    n = len(own)
    predictor = get_predictor()
    # This fits only the support-conditioned kernel, without item difficulty
    # or a subject-ability model. Its existing acquisition geometry is retained.
    kernel = float(predictor.predict(input, evidence))
    base, residual = predictor.initial(subject), 0.
    if n:
        positions, state = next(reversed(predictor.fits.values()))
        position = positions[subject_key(subject)]
        base = float(state['base'][position])
        features = predictor.features([item])
        semantic, lexical = transform(state['geometry'], features[:, :384], features[:, 384:])
        similarity = (semantic @ state['semantic'].T + lexical @ state['lexical'].T) / 2
        cross = np.exp(np.minimum(similarity - 1, 0) / state['bandwidth'])
        residual = float((cross @ state['dual'])[0] - state['bias'][position])
    if abs(np.clip(base + residual, 0, 1) - kernel) > 1e-12:
        raise RuntimeError('Kernel state does not match the current evidence.')
    delta = _calibration_adjust(.5, item, _embedding(item.get('item_content', '')),
                                n, _resources()['topic']) - .5
    return {key: np.asarray(value) for key, value in {
        'n': n, 's': sum(y for _, y in own),
        'prior': _query_prior(input),
        'support_prior': float(np.mean([_query_prior(x) for x, _ in own])) if own else .5,
        'kernel_base': base, 'residual': residual, 'topic': delta,
    }.items()}


def predict(input: list, labeled: list | None = None) -> float:
    return float(probabilities(prediction_parts(input, labeled), _resources()['config']))
