"""Adaptive covariance acquisition for the kernel Probe."""

from threadpoolctl import threadpool_limits

from kernel_probe import get_predictor
from project_policy import make_policy
from project_core import local_gram

_policy = None


def acquisition_function(input: list[dict], labeled: list | None = None,
                         context: dict | None = None) -> bool:
    """Select candidates with the validation-selected adaptive_4_warm rule."""
    global _policy
    if context is None:
        raise ValueError("Streaming acquisition requires evaluator context")
    if _policy is None:
        predictor = get_predictor()

        def gram(items):
            return local_gram(predictor.features(items))

        _policy = make_policy(gram, predictor.config)
    with threadpool_limits(limits=2):
        return _policy.choose(input, labeled or [], context)
