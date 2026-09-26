"""Classification accuracy."""

from __future__ import annotations

from decisionops.schemas import DecisionCase, DecisionPrediction


def accuracy(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> float:
    if not pairs:
        raise ValueError("cannot compute accuracy on empty input")
    correct = sum(1 for case, pred in pairs if pred.prediction == case.label)
    return correct / len(pairs)
