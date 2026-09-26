"""Apply calibration-chosen thresholds on a held-out set."""

from __future__ import annotations

from decisionops.schemas import DecisionCase, DecisionPrediction, ThresholdRecommendation
from decisionops.selective.threshold import evaluate_threshold


def evaluate_fixed_thresholds(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    *,
    target_risks_and_thresholds: list[tuple[float, float]],
) -> list[ThresholdRecommendation]:
    """
    Evaluate pre-selected thresholds on `pairs` (typically the test split).

    `target_risks_and_thresholds` is a list of (target_risk, threshold) chosen
    on the calibration split.
    """
    out: list[ThresholdRecommendation] = []
    for target_risk, threshold in target_risks_and_thresholds:
        point = evaluate_threshold(pairs, threshold)
        sel_acc = 0.0 if point.n_accepted == 0 else float(point.selective_accuracy)
        sel_risk = 1.0 if point.n_accepted == 0 else float(point.selective_risk)
        out.append(
            ThresholdRecommendation(
                target_risk=float(target_risk),
                threshold=float(threshold),
                coverage=point.coverage,
                selective_accuracy=sel_acc,
                selective_risk=sel_risk,
                n_accepted=point.n_accepted,
                n_total=point.n_total,
            )
        )
    return out