"""
Offline cascade / cost simulation.

Given a decision contract with per-choice costs and a labeled prediction pool,
sweep confidence thresholds and report the trade-off between money saved
(versus always escalating) and the joint automated miss rate.

Escalation cost is *oracle*: when the true label is an escalate choice we pay
that label's cost; when the label was automatic but we still escalate, we pay
the cheapest escalate choice. This is an optimistic bound on cascade cost —
real multi-stage routing comes later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from decisionops.contract import DecisionContract
from decisionops.io import align_cases_and_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction, SplitInfo
from decisionops.selective.threshold import risk_coverage_curve
from decisionops.split import SplitResult, split_pairs


class CostPoint(BaseModel):
    """One operating point on the cost–miss curve."""

    threshold: float
    n_total: int
    n_auto: int
    automation_rate: float
    """Share of traffic executed automatically (coverage)."""

    n_miss: int
    miss_rate: float
    """Joint automated error rate: auto-executed AND wrong, over all traffic."""

    selective_risk: float | None = None
    """Error rate among the auto-executed subset; None when nothing is auto."""

    mean_cost: float
    """Average cost paid at this threshold (same unit as contract.costs)."""

    cost_saved: float
    """mean(always-escalate) − mean_cost. Positive = cheaper than always escalate."""

    cost_saved_frac: float | None = None
    """cost_saved / mean(always-escalate); None when the baseline is 0."""


class CascadeReport(BaseModel):
    split: SplitInfo
    provider: str | None = None
    model: str | None = None
    contract: dict[str, Any]
    baseline_mean_cost: float
    """Mean cost of escalating every case (oracle escalate path)."""

    always_auto_mean_cost: float
    """Mean cost of auto-executing every prediction (no threshold)."""

    always_auto_miss_rate: float
    """Joint miss rate with no threshold (every eligible prediction accepted)."""

    curve: list[CostPoint]
    """Threshold sweep on the evaluation split, sorted by increasing threshold."""

    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _require_costs(contract: DecisionContract) -> None:
    missing = [c for c in contract.choices if c not in contract.costs]
    if missing:
        raise ValueError(
            f"contract {contract.name!r} is missing costs for {missing}; "
            "cascade needs a cost for every choice"
        )
    if not contract.escalate_choices:
        raise ValueError(
            f"contract {contract.name!r} has no escalate choices; "
            "cascade needs at least one non-auto path to price against"
        )


def choice_cost(contract: DecisionContract, choice: str) -> float:
    return float(contract.costs[choice])


def escalate_cost(contract: DecisionContract, case: DecisionCase) -> float:
    """
    Oracle cost of sending `case` down an escalate path.

    Prefer the true label when it is itself an escalate choice; otherwise pay the
    cheapest escalate choice (unnecessary review of an auto-safe case).
    """
    if case.label not in contract.auto_set:
        return choice_cost(contract, case.label)
    return min(choice_cost(contract, c) for c in contract.escalate_choices)


def auto_cost(contract: DecisionContract, prediction: DecisionPrediction) -> float:
    return choice_cost(contract, prediction.prediction)


def _point_at_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    threshold: float,
    *,
    contract: DecisionContract,
    baseline_mean: float,
) -> CostPoint:
    auto = contract.auto_set
    n = len(pairs)
    costs = np.empty(n, dtype=np.float64)
    is_auto = np.zeros(n, dtype=bool)
    is_miss = np.zeros(n, dtype=bool)

    for i, (case, pred) in enumerate(pairs):
        eligible = pred.prediction in auto and pred.confidence >= threshold
        if eligible:
            is_auto[i] = True
            costs[i] = auto_cost(contract, pred)
            if pred.prediction != case.label:
                is_miss[i] = True
        else:
            costs[i] = escalate_cost(contract, case)

    n_auto = int(is_auto.sum())
    n_miss = int(is_miss.sum())
    mean_cost = float(costs.mean())
    saved = baseline_mean - mean_cost
    saved_frac = (saved / baseline_mean) if baseline_mean > 0 else None
    sel_risk = float(is_miss[is_auto].mean()) if n_auto else None

    return CostPoint(
        threshold=float(threshold),
        n_total=n,
        n_auto=n_auto,
        automation_rate=n_auto / n,
        n_miss=n_miss,
        miss_rate=n_miss / n,
        selective_risk=sel_risk,
        mean_cost=mean_cost,
        cost_saved=saved,
        cost_saved_frac=saved_frac,
    )


def cost_curve(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    contract: DecisionContract,
    *,
    thresholds: list[float] | None = None,
) -> tuple[float, float, float, list[CostPoint]]:
    """
    Sweep thresholds and return (baseline_mean, always_auto_mean, always_auto_miss, curve).
    """
    if not pairs:
        raise ValueError("cannot build a cost curve on an empty set")
    _require_costs(contract)

    baseline = float(np.mean([escalate_cost(contract, case) for case, _ in pairs]))

    # Always-auto: accept every prediction that lands on an auto choice, else escalate.
    always = _point_at_threshold(pairs, threshold=0.0, contract=contract, baseline_mean=baseline)

    if thresholds is None:
        # Reuse the selective grid so the cost curve lines up with risk-coverage.
        thresholds = [
            p.threshold
            for p in risk_coverage_curve(pairs, auto_choices=contract.auto_set)
        ]

    curve = [
        _point_at_threshold(pairs, t, contract=contract, baseline_mean=baseline)
        for t in thresholds
    ]
    return baseline, always.mean_cost, always.miss_rate, curve


def run_cascade(
    cases: list[DecisionCase],
    predictions: list[DecisionPrediction],
    contract: DecisionContract,
    *,
    cal_ratio: float = 0.3,
    seed: int = 42,
) -> CascadeReport:
    """
    Offline cascade simulation on a held-out split.

    Calibration is reserved for a future CRC overlay; the cost curve itself is
    scored on the test split so it is not fit to the numbers it reports.
    """
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError(f"cal_ratio must be in (0, 1), got {cal_ratio}")
    contract.validate_data(cases, predictions)
    _require_costs(contract)

    pairs = align_cases_and_predictions(cases, predictions)
    split: SplitResult = split_pairs(pairs, cal_ratio=cal_ratio, seed=seed)
    eval_pairs = split.test

    baseline, always_cost, always_miss, curve = cost_curve(eval_pairs, contract)

    preds_all = [p for _, p in pairs]
    providers = {p.provider for p in preds_all if p.provider}
    models = {p.model for p in preds_all if p.model}

    notes = [
        "Escalation costs are oracle (true escalate label, else cheapest escalate).",
        "miss_rate is the joint automated error rate (auto AND wrong over all traffic), "
        "the same quantity CRC certifies.",
        "Curve is scored on the held-out test split; thresholds are not CRC-certified.",
    ]
    if baseline <= 0:
        notes.append(
            "Baseline escalate cost is 0 — cost_saved_frac is undefined; "
            "check that escalate choices have positive costs."
        )

    return CascadeReport(
        split=SplitInfo(
            seed=split.seed,
            cal_ratio=split.cal_ratio,
            n_total=split.n_total,
            n_calibration=split.n_calibration,
            n_test=split.n_test,
        ),
        provider=next(iter(providers)) if len(providers) == 1 else None,
        model=next(iter(models)) if len(models) == 1 else None,
        contract=contract.summary(),
        baseline_mean_cost=baseline,
        always_auto_mean_cost=always_cost,
        always_auto_miss_rate=always_miss,
        curve=curve,
        notes=notes,
        metadata={
            "protocol": "cal_test_cost_curve",
            "escalation_pricing": "oracle",
            "n_thresholds": len(curve),
        },
    )
