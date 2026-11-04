"""
Certified two-stage cascade thresholds (Learn-Then-Test on a 2-D grid).

Two decision models in series, then the expensive path:

    stage 1   M1 picks an automatic choice and conf1 >= lambda1  ->  AUTO (M1)
    stage 2   else M2 picks an automatic choice and conf2 >= lambda2  ->  AUTO (M2)
    else      ESCALATE

The certified quantity is the **joint automated error rate**: the share of all
traffic auto-executed by either stage AND wrong. lambda = None disables a
stage, so a single-model certificate is the corner `lambda1 = None`.

**Why not CRC.** Risk is monotone in lambda2 but not in lambda1: raising
lambda1 hands cases from M1 to M2, which may accept them and be wrong more
often. CRC needs a monotone loss, so this uses Learn-Then-Test instead and is
high-probability only.

**Procedure.** Both grids are fixed before any calibration label is seen
(quantiles of confidences on the temperature-fit slice, which is disjoint from
the calibration set). For each of the m values of lambda1 (a row, including
"stage 1 off"), lambda2 is scanned from strict (stage 2 off) to loose with
fixed-sequence testing at level delta/m; the scan stops at the first failure.
Rows are combined by Bonferroni. Each hypothesis

    H0(l1, l2): joint error(l1, l2) > alpha

is rejected when the one-sided Clopper-Pearson upper limit on k errors in n
calibration cases is <= alpha (exact: k ~ Bin(n, R) under 0-1 loss). Then

    P( every passed (l1, l2) has true joint error <= alpha ) >= 1 - delta

over the calibration draw, so choosing the cheapest passed point keeps the
guarantee. The price is the delta/m level: the zero-error floor is
`ln(delta/m) / ln(1 - alpha)` — 95 at alpha=5%, 90%, m=13 (single model: 45).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from decisionops.conformal.risk_control import (
    _check_confidence_level,
    clopper_pearson_upper,
    min_calibration_samples_hp,
)
from decisionops.schemas import DecisionCase, DecisionPrediction

CASCADE_QUANTITY = "cascade_joint_automated_error_rate"
Objective = Literal["cost", "automation"]


@dataclass(frozen=True)
class StageArrays:
    """
    Per-case view of one stage on a fixed case order.

    `confidence` is -inf where the stage's prediction is not an automatic
    choice, so that stage never accepts the case at any threshold.
    """

    confidence: np.ndarray
    wrong: np.ndarray
    """1.0 where the stage's prediction differs from the label, else 0.0."""

    auto_cost: np.ndarray
    """Path cost if this stage auto-executes the case (0 when unpriced)."""


def quantile_grid(values: Sequence[float] | np.ndarray, size: int) -> list[float]:
    """
    `size` interior quantiles of `values`, deduplicated, ascending.

    Fixed before calibration labels are seen: callers pass confidences from a
    slice disjoint from the calibration set. Empty input falls back to an
    evenly spaced grid on (0, 1).
    """
    if size < 1:
        raise ValueError(f"grid size must be >= 1, got {size}")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    levels = np.linspace(0.0, 1.0, size + 2)[1:-1]
    if arr.size == 0:
        return [float(v) for v in levels]
    return sorted({float(v) for v in np.quantile(arr, levels)})


def min_cascade_calibration(alpha: float, confidence_level: float, n_rows: int) -> int:
    """Zero-error floor at the per-row level delta/m."""
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows}")
    _check_confidence_level(confidence_level)
    delta = 1.0 - confidence_level
    return min_calibration_samples_hp(alpha, 1.0 - delta / n_rows)


def _accept(conf: np.ndarray, lam: float | None) -> np.ndarray:
    if lam is None:
        return np.zeros(conf.shape, dtype=bool)
    return conf >= lam


def cascade_routes(
    s1: StageArrays, s2: StageArrays, lambda1: float | None, lambda2: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (auto at stage 1, auto at stage 2) for one threshold pair."""
    acc1 = _accept(s1.confidence, lambda1)
    acc2 = ~acc1 & _accept(s2.confidence, lambda2)
    return acc1, acc2


def stage_arrays_from_pairs(
    pairs: Sequence[tuple[DecisionCase, DecisionPrediction]],
    *,
    auto_choices: frozenset[str] | None,
    path_costs: dict[str, float] | None = None,
) -> StageArrays:
    """
    Build stage arrays from aligned (case, prediction) pairs.

    Non-automatic predictions get confidence -inf so they never auto-execute.
    `path_costs` is the contract cost of executing the predicted choice.
    """
    n = len(pairs)
    conf = np.empty(n, dtype=np.float64)
    wrong = np.empty(n, dtype=np.float64)
    auto_cost = np.zeros(n, dtype=np.float64)
    for i, (case, pred) in enumerate(pairs):
        eligible = auto_choices is None or pred.prediction in auto_choices
        conf[i] = pred.confidence if eligible else -np.inf
        wrong[i] = 0.0 if pred.prediction == case.label else 1.0
        if path_costs is not None:
            auto_cost[i] = float(path_costs.get(pred.prediction, 0.0))
    return StageArrays(confidence=conf, wrong=wrong, auto_cost=auto_cost)


class CascadePoint(BaseModel):
    """One threshold pair measured on a set of cases."""

    lambda1: float | None
    lambda2: float | None
    n_total: int
    n_auto_stage1: int
    n_auto_stage2: int
    n_errors: int
    joint_error_rate: float
    automation_rate: float
    stage2_call_rate: float
    """Share of traffic M2 is called on (M1 did not auto-execute; 0 when stage 2 is off)."""

    mean_cost: float | None = None
    upper_bound: float | None = None
    """Clopper-Pearson limit at the per-row level; set on calibration points only."""


def measure_cascade(
    s1: StageArrays,
    s2: StageArrays,
    lambda1: float | None,
    lambda2: float | None,
    *,
    stage_costs: tuple[float, float] | None = None,
    escalate_costs: np.ndarray | None = None,
) -> CascadePoint:
    """
    Route every case with (lambda1, lambda2) and count what happened.

    Cost is reported when `stage_costs` and `escalate_costs` are given: every
    case pays M1, cases M1 does not auto-execute also pay M2, then the path
    cost (auto choice cost, or the oracle escalation cost). A disabled stage
    (lambda None) is never called, so it costs nothing.
    """
    n = int(s1.confidence.size)
    acc1, acc2 = cascade_routes(s1, s2, lambda1, lambda2)
    errors = int(np.sum(s1.wrong[acc1]) + np.sum(s2.wrong[acc2]))
    reach2 = ~acc1 if lambda2 is not None else np.zeros(n, dtype=bool)
    mean_cost = None
    if stage_costs is not None and escalate_costs is not None:
        c1, c2 = stage_costs
        esc = ~acc1 & ~acc2
        per_case = (
            (c1 if lambda1 is not None else 0.0)
            + c2 * reach2
            + np.where(acc1, s1.auto_cost, 0.0)
            + np.where(acc2, s2.auto_cost, 0.0)
            + np.where(esc, escalate_costs, 0.0)
        )
        mean_cost = float(per_case.mean())
    n1, n2 = int(acc1.sum()), int(acc2.sum())
    return CascadePoint(
        lambda1=lambda1,
        lambda2=lambda2,
        n_total=n,
        n_auto_stage1=n1,
        n_auto_stage2=n2,
        n_errors=errors,
        joint_error_rate=errors / n,
        automation_rate=(n1 + n2) / n,
        stage2_call_rate=float(reach2.mean()),
        mean_cost=mean_cost,
    )


class CascadeRow(BaseModel):
    """Fixed-sequence scan of lambda2 for one lambda1."""

    lambda1: float | None
    n_passed: int
    loosest_lambda2: float | None = None
    """Loosest passed lambda2; None when only "stage 2 off" passed (or nothing)."""


class CascadeCertificate(BaseModel):
    """Outcome of certifying a two-stage cascade at one alpha."""

    alpha: float
    guarantee: Literal["high_probability"] = "high_probability"
    confidence_level: float
    controlled_quantity: str = CASCADE_QUANTITY
    feasible: bool
    """True when some passed point auto-executes at least one calibration case."""

    lambda1: float | None = None
    lambda2: float | None = None
    objective: Objective
    n_calibration: int
    n_rows: int
    """m: number of lambda1 values, the Bonferroni divisor."""

    per_row_level: float
    """1 - delta/m, the level of each Clopper-Pearson test."""

    min_n_required: int
    calibration: CascadePoint | None = None
    """The chosen point measured on calibration (cost there is optimistic)."""

    n_passed: int = 0
    rows: list[CascadeRow] = Field(default_factory=list)
    lambda1_grid: list[float] = Field(default_factory=list)
    lambda2_grid: list[float] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _better(a: CascadePoint, b: CascadePoint | None, objective: Objective) -> bool:
    """Is `a` preferred over the incumbent `b`?"""
    if b is None:
        return True
    if objective == "cost":
        assert a.mean_cost is not None and b.mean_cost is not None
        key_a = (a.mean_cost, -a.automation_rate, a.n_errors)
        key_b = (b.mean_cost, -b.automation_rate, b.n_errors)
    else:
        key_a = (-a.automation_rate, a.n_errors, a.stage2_call_rate)
        key_b = (-b.automation_rate, b.n_errors, b.stage2_call_rate)
    return key_a < key_b


def certify_cascade(
    s1: StageArrays,
    s2: StageArrays,
    alpha: float,
    *,
    lambda1_grid: Sequence[float],
    lambda2_grid: Sequence[float],
    confidence_level: float = 0.9,
    stage_costs: tuple[float, float] | None = None,
    escalate_costs: np.ndarray | None = None,
) -> CascadeCertificate:
    """
    Learn-Then-Test over (lambda1, lambda2); pick the best passed point.

    `s1` / `s2` must describe the **calibration** cases only, in the same order.
    Grids must be fixed without calibration labels. "Stage off" (None) is added
    to both grids, so m = len(lambda1_grid) + 1.

    The objective is mean calibration cost when `stage_costs` and
    `escalate_costs` are given, else automation rate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    _check_confidence_level(confidence_level)
    n = int(s1.confidence.size)
    if n == 0:
        raise ValueError("cannot certify on empty calibration set")
    if s2.confidence.size != n:
        raise ValueError("stage arrays must cover the same calibration cases")
    for w in (s1.wrong, s2.wrong):
        if not np.all((w == 0.0) | (w == 1.0)):
            raise ValueError("cascade certification requires 0-1 loss")
    priced = stage_costs is not None and escalate_costs is not None
    objective: Objective = "cost" if priced else "automation"

    grid1: list[float | None] = [None, *sorted(set(map(float, lambda1_grid)))]
    # Strict -> loose: stage 2 off first, then descending thresholds.
    grid2: list[float | None] = [None, *sorted(set(map(float, lambda2_grid)), reverse=True)]
    m = len(grid1)
    delta = 1.0 - confidence_level
    level = 1.0 - delta / m
    min_n = min_cascade_calibration(alpha, confidence_level, m)
    base: dict[str, Any] = {
        "alpha": alpha,
        "confidence_level": confidence_level,
        "objective": objective,
        "n_calibration": n,
        "n_rows": m,
        "per_row_level": level,
        "min_n_required": min_n,
        "lambda1_grid": [g for g in grid1 if g is not None],
        "lambda2_grid": sorted(g for g in grid2 if g is not None),
    }
    if clopper_pearson_upper(0, n, level) > alpha:
        return CascadeCertificate(
            feasible=False,
            notes=[
                f"Infeasible: even with zero errors the {level:.2%} upper bound on n={n} "
                f"exceeds alpha={alpha:.4f}. Need at least {min_n} calibration samples "
                f"(m={m} rows, delta={delta:g})."
            ],
            **base,
        )

    chosen: CascadePoint | None = None
    rows: list[CascadeRow] = []
    n_passed = 0
    for l1 in grid1:
        passed_here = 0
        loosest: float | None = None
        for l2 in grid2:
            point = measure_cascade(
                s1, s2, l1, l2, stage_costs=stage_costs, escalate_costs=escalate_costs
            )
            ucb = clopper_pearson_upper(point.n_errors, n, level)
            if ucb > alpha:
                break
            point.upper_bound = ucb
            passed_here += 1
            loosest = l2
            if point.automation_rate > 0 and _better(point, chosen, objective):
                chosen = point
        n_passed += passed_here
        rows.append(CascadeRow(lambda1=l1, n_passed=passed_here, loosest_lambda2=loosest))

    notes: list[str] = []
    if chosen is None:
        notes.append(
            "No passed threshold pair auto-executes any calibration case; escalate all "
            "traffic at this alpha."
        )
        return CascadeCertificate(
            feasible=False, n_passed=n_passed, rows=rows, notes=notes, **base
        )

    if chosen.lambda1 is None:
        notes.append("Best certified point does not use stage 1 (M2 alone).")
    elif chosen.lambda2 is None:
        notes.append("Best certified point does not use stage 2 (M1 alone).")
    notes.append(
        f"Cascade certificate: with probability >= {confidence_level:.0%} over the "
        "calibration draw, the true joint automated error rate of every passed "
        "threshold pair, including the chosen one, is <= alpha."
    )
    if objective == "cost":
        notes.append("Calibration cost is optimistic (the point was chosen on it).")
    return CascadeCertificate(
        feasible=True,
        lambda1=chosen.lambda1,
        lambda2=chosen.lambda2,
        calibration=chosen,
        n_passed=n_passed,
        rows=rows,
        notes=notes,
        **base,
    )
