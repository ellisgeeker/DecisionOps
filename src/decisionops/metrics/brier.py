"""Multiclass Brier score."""

from __future__ import annotations

import numpy as np

from decisionops.schemas import DecisionCase, DecisionPrediction


def brier_score(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> float:
    """
    Multiclass Brier score:

        mean_i sum_k (p_ik - y_ik)^2

    Lower is better. Perfect deterministic correct predictions score 0.
    """
    if not pairs:
        raise ValueError("cannot compute brier score on empty input")

    scores: list[float] = []
    for case, pred in pairs:
        labels = list(pred.probabilities.keys())
        # One-hot over the model's support; unknown gold label => all zeros + miss.
        y = np.array([1.0 if lab == case.label else 0.0 for lab in labels], dtype=np.float64)
        p = np.array([pred.probabilities[lab] for lab in labels], dtype=np.float64)
        # If gold label is outside predicted support, add the missing one-hot mass.
        if case.label not in pred.probabilities:
            # y has all zeros above; squared error for missing class is 1, plus p norms.
            scores.append(float(np.sum((p - y) ** 2) + 1.0))
        else:
            scores.append(float(np.sum((p - y) ** 2)))
    return float(np.mean(scores))
