"""Negative log-likelihood of the gold label."""

from __future__ import annotations

import math

from decisionops.schemas import DecisionCase, DecisionPrediction

_EPS = 1e-12


def negative_log_likelihood(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> float:
    if not pairs:
        raise ValueError("cannot compute nll on empty input")
    total = 0.0
    for case, pred in pairs:
        p = pred.probability_for(case.label, default=0.0)
        total += -math.log(max(p, _EPS))
    return total / len(pairs)
