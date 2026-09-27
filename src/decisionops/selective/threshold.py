"""Threshold sweep and target-risk recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decisionops.schemas import DecisionCase, DecisionPrediction, ThresholdRecommendation


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    selective_accuracy: float
    selective_risk: float
    n_accepted: int
    n_total: int


def _confidence_and_correct(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([pred.confidence for _, pred in pairs], dtype=np.float64)
    correct = np.array(
        [1.0 if pred.prediction == case.label else 0.0 for case, pred in pairs],
        dtype=np.float64,
    )
    return conf, correct


def auto_eligible(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    auto_choices: frozenset[str] | None,
) -> np.ndarray:
    """
    Which cases *could* be auto-executed: the predicted choice is on the cheap path.

    With no contract (`auto_choices=None`) every prediction is eligible, which is
    the original symmetric behaviour. With a contract, a prediction of an
    escalation choice is never auto-accepted, so its errors are not risk.
    """
    if auto_choices is None:
        return np.ones(len(pairs), dtype=bool)
    return np.array([pred.prediction in auto_choices for _, pred in pairs], dtype=bool)


def evaluate_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    threshold: float,
    *,
    auto_choices: frozenset[str] | None = None,
) -> RiskCoveragePoint:
    """
    Accept when the prediction is auto-eligible and max probability >= threshold.

    Coverage is the share of all cases executed automatically; selective risk is
    the error rate among those. Escalated cases never count as errors.
    """
    if not pairs:
        raise ValueError("cannot evaluate threshold on empty input")
    conf, correct = _confidence_and_correct(pairs)
    accepted = auto_eligible(pairs, auto_choices) & (conf >= threshold)
    n_total = len(pairs)
    n_accepted = int(accepted.sum())
    if n_accepted == 0:
        return RiskCoveragePoint(
            threshold=float(threshold),
            coverage=0.0,
            selective_accuracy=float("nan"),
            selective_risk=float("nan"),
            n_accepted=0,
            n_total=n_total,
        )
    sel_acc = float(correct[accepted].mean())
    return RiskCoveragePoint(
        threshold=float(threshold),
        coverage=n_accepted / n_total,
        selective_accuracy=sel_acc,
        selective_risk=1.0 - sel_acc,
        n_accepted=n_accepted,
        n_total=n_total,
    )


def risk_coverage_curve(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    thresholds: list[float] | None = None,
    *,
    auto_choices: frozenset[str] | None = None,
) -> list[RiskCoveragePoint]:
    """Sweep confidence thresholds and compute risk/coverage points."""
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.01)] + [0.99, 1.0]
        # unique + sorted
        thresholds = sorted(set(float(t) for t in thresholds))
    return [evaluate_threshold(pairs, t, auto_choices=auto_choices) for t in thresholds]


def recommend_thresholds(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    target_risks: list[float] | None = None,
    *,
    auto_choices: frozenset[str] | None = None,
) -> list[ThresholdRecommendation]:
    """
    For each target risk α, pick the *lowest* threshold whose empirical
    selective risk on this set is <= α (maximizes coverage under the constraint).

    Note: this is an empirical operating point on the evaluation set, not a
    conformal certificate; `decisionops.conformal` provides the certified path.
    """
    if target_risks is None:
        target_risks = [0.01, 0.05, 0.10]

    curve = risk_coverage_curve(pairs, auto_choices=auto_choices)
    # Sort by increasing threshold so we can scan for feasible points.
    curve_sorted = sorted(curve, key=lambda p: p.threshold)

    recommendations: list[ThresholdRecommendation] = []
    for alpha in target_risks:
        feasible = [
            p
            for p in curve_sorted
            if p.n_accepted > 0 and not np.isnan(p.selective_risk) and p.selective_risk <= alpha
        ]
        if not feasible:
            # No threshold meets the risk target; report the most conservative point.
            best = curve_sorted[-1]
            recommendations.append(
                ThresholdRecommendation(
                    target_risk=float(alpha),
                    threshold=best.threshold,
                    coverage=best.coverage,
                    selective_accuracy=(
                        best.selective_accuracy if not np.isnan(best.selective_accuracy) else 0.0
                    ),
                    selective_risk=(
                        best.selective_risk if not np.isnan(best.selective_risk) else 1.0
                    ),
                    n_accepted=best.n_accepted,
                    n_total=best.n_total,
                )
            )
            continue

        # Among feasible points, maximize coverage (equivalently minimize threshold).
        best = max(feasible, key=lambda p: (p.coverage, -p.threshold))
        recommendations.append(
            ThresholdRecommendation(
                target_risk=float(alpha),
                threshold=best.threshold,
                coverage=best.coverage,
                selective_accuracy=best.selective_accuracy,
                selective_risk=best.selective_risk,
                n_accepted=best.n_accepted,
                n_total=best.n_total,
            )
        )
    return recommendations
