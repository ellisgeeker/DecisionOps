"""
Certified two-stage cascade (Learn-Then-Test on a 2-D grid).

Analytic pool: both stages are calibrated and independent. Stage s has
confidence c ~ U(LO_s, HI_s) and P(correct | c) = c, so accepting c >= l
covers p_s(l) = (HI_s - l) / (HI_s - LO_s) of traffic with error rate
e_s(l) = 1 - (l + HI_s) / 2. Stage 2 sees only what stage 1 did not accept:

    true joint error(l1, l2) = p1 e1 + (1 - p1) p2 e2

The guarantee is checked against that closed form, not a noisy test split.
"""

from __future__ import annotations

import numpy as np
import pytest

import decisionops.conformal.cascade_risk as cascade_risk
from decisionops.conformal.cascade_risk import (
    CASCADE_QUANTITY,
    StageArrays,
    certify_cascade,
    measure_cascade,
    min_cascade_calibration,
    quantile_grid,
)
from decisionops.conformal.risk_control import clopper_pearson_upper

LO1, HI1 = 0.55, 0.999
LO2, HI2 = 0.70, 0.999
GRID_SIZE = 12


def _tail(lam, lo, hi):
    if lam is None:
        return 0.0, 0.0
    lam = min(max(lam, lo), hi)
    return (hi - lam) / (hi - lo), 1.0 - (lam + hi) / 2.0


def true_risk(l1, l2) -> float:
    p1, e1 = _tail(l1, LO1, HI1)
    p2, e2 = _tail(l2, LO2, HI2)
    return p1 * e1 + (1.0 - p1) * p2 * e2


def _stages(rng: np.random.Generator, n: int) -> tuple[StageArrays, StageArrays]:
    zero = np.zeros(n)
    c1 = rng.uniform(LO1, HI1, n)
    c2 = rng.uniform(LO2, HI2, n)
    w1 = (rng.random(n) >= c1).astype(float)
    w2 = (rng.random(n) >= c2).astype(float)
    return StageArrays(c1, w1, zero), StageArrays(c2, w2, zero)


def _grids() -> tuple[list[float], list[float]]:
    # Fixed before any calibration label: quantiles of an independent draw.
    rng = np.random.default_rng(1)
    return (
        quantile_grid(rng.uniform(LO1, HI1, 500), GRID_SIZE),
        quantile_grid(rng.uniform(LO2, HI2, 500), GRID_SIZE),
    )


def _violation_rate(alpha, level, n, reps, seed0):
    g1, g2 = _grids()
    viol = feasible = 0
    for rep in range(reps):
        s1, s2 = _stages(np.random.default_rng(seed0 + rep), n)
        cert = certify_cascade(
            s1, s2, alpha, lambda1_grid=g1, lambda2_grid=g2, confidence_level=level
        )
        if cert.feasible:
            feasible += 1
            viol += true_risk(cert.lambda1, cert.lambda2) > alpha
    return viol, feasible


def test_floor_pays_for_the_rows():
    # ln(delta/m) / ln(1 - alpha): Bonferroni over 13 rows roughly doubles the floor.
    assert min_cascade_calibration(0.05, 0.9, 1) == 45
    assert min_cascade_calibration(0.05, 0.9, 13) == 95
    assert min_cascade_calibration(0.01, 0.9, 13) == 485


def test_violation_rate_is_at_most_delta():
    """Measured over 600 reps: 1.0% of certificates above alpha, budget 20%."""
    alpha, level = 0.05, 0.8
    viol, feasible = _violation_rate(alpha, level, n=300, reps=600, seed0=60_000)
    assert feasible >= 500
    assert viol / feasible <= 1.0 - level, (viol, feasible)


def test_dropping_bonferroni_breaks_the_guarantee(monkeypatch):
    """
    Teeth for the delta/m row level: testing every row at delta lets the scan
    pick from many correlated rows. Measured 141/600 = 23.5% violations against
    the 20% budget (the real rule: 1.0%). The margin is ~2 SE; the seeds are
    fixed, so the result is deterministic.
    """
    alpha, level = 0.05, 0.8
    real = cascade_risk.clopper_pearson_upper
    monkeypatch.setattr(
        cascade_risk, "clopper_pearson_upper", lambda k, n, _lv: real(k, n, level)
    )
    viol, feasible = _violation_rate(alpha, level, n=300, reps=600, seed0=60_000)
    assert viol / feasible > 1.0 - level, (viol, feasible)


def test_picking_by_empirical_risk_breaks_the_guarantee():
    """
    The rule CRC-free tooling would use: most automation among points whose
    calibration error is <= alpha. Measured 56% violations at alpha=5%.
    """
    alpha, level = 0.05, 0.9
    g1, g2 = _grids()
    viol = 0
    reps = 200
    for rep in range(reps):
        s1, s2 = _stages(np.random.default_rng(50_000 + rep), 600)
        best = None
        for l1 in [None, *g1]:
            for l2 in [None, *g2]:
                p = measure_cascade(s1, s2, l1, l2)
                eligible = p.joint_error_rate <= alpha and p.automation_rate > 0
                if eligible and (best is None or p.automation_rate > best.automation_rate):
                    best = p
        viol += true_risk(best.lambda1, best.lambda2) > alpha
    assert viol / reps > 1.0 - level


def test_risk_is_not_monotone_in_lambda1():
    """Why CRC does not apply: raising lambda1 hands cases to a worse stage 2."""
    zero = np.zeros(4)
    # Stage 1 is right on everything it is confident about; stage 2 is always wrong.
    s1 = StageArrays(np.array([0.9, 0.8, 0.7, 0.6]), np.zeros(4), zero)
    s2 = StageArrays(np.array([0.9, 0.9, 0.9, 0.9]), np.ones(4), zero)
    loose = measure_cascade(s1, s2, 0.6, 0.5)
    strict = measure_cascade(s1, s2, 0.85, 0.5)
    assert loose.n_errors == 0
    assert strict.n_errors == 3


def test_stage1_off_row_is_fixed_sequence_on_m2_alone():
    """
    Nesting: the lambda1=None row is M2 alone, scanned strict -> loose on the
    lambda2 grid at the per-row level 1 - delta/m.
    """
    s1, s2 = _stages(np.random.default_rng(70_000), 800)
    g1, g2 = _grids()
    level = 0.9
    cert = certify_cascade(s1, s2, 0.05, lambda1_grid=g1, lambda2_grid=g2, confidence_level=level)
    row = cert.rows[0]
    assert row.lambda1 is None

    per_row = 1.0 - (1.0 - level) / (len(g1) + 1)
    assert cert.per_row_level == pytest.approx(per_row)
    loosest, passed = None, 1  # "stage 2 off" makes no errors and always passes
    for lam in sorted(g2, reverse=True):
        k = int(s2.wrong[s2.confidence >= lam].sum())
        if clopper_pearson_upper(k, 800, per_row) > 0.05:
            break
        loosest, passed = lam, passed + 1
    assert row.loosest_lambda2 == loosest
    assert row.n_passed == passed


def test_chosen_point_passed_and_beats_both_single_stage_rows():
    s1, s2 = _stages(np.random.default_rng(71_000), 2000)
    g1, g2 = _grids()
    cert = certify_cascade(s1, s2, 0.05, lambda1_grid=g1, lambda2_grid=g2, confidence_level=0.9)
    assert cert.feasible
    assert cert.controlled_quantity == CASCADE_QUANTITY
    assert cert.calibration.upper_bound <= 0.05

    # Objective is automation: no single-stage passed point automates more.
    m2_alone = measure_cascade(s1, s2, None, cert.rows[0].loosest_lambda2)
    assert cert.calibration.automation_rate >= m2_alone.automation_rate
    m1_alone = [
        measure_cascade(s1, s2, row.lambda1, None) for row in cert.rows[1:] if row.n_passed
    ]
    assert all(cert.calibration.automation_rate >= p.automation_rate for p in m1_alone)
    # On this pool the two stages together beat either alone.
    assert cert.lambda1 is not None and cert.lambda2 is not None


def test_cost_objective_counts_stage2_calls():
    """With prices, every case pays M1; only cases M1 does not take pay M2."""
    n = 4
    s1 = StageArrays(np.array([0.9, 0.9, 0.1, 0.1]), np.zeros(n), np.full(n, 0.5))
    s2 = StageArrays(np.array([0.9, 0.1, 0.9, 0.1]), np.zeros(n), np.full(n, 0.5))
    esc = np.full(n, 10.0)
    point = measure_cascade(
        s1, s2, 0.5, 0.5, stage_costs=(1.0, 2.0), escalate_costs=esc
    )
    # M1 all 4 (1.0) + M2 on 2 (2.0) + auto paths for 3 (0.5) + escalate 1 (10)
    assert point.mean_cost == pytest.approx((4 * 1.0 + 2 * 2.0 + 3 * 0.5 + 10.0) / 4)
    assert point.stage2_call_rate == 0.5
    assert (point.n_auto_stage1, point.n_auto_stage2) == (2, 1)

    # A disabled stage is never called: stage 2 off pays M1 + paths only.
    off = measure_cascade(s1, s2, 0.5, None, stage_costs=(1.0, 2.0), escalate_costs=esc)
    assert off.stage2_call_rate == 0.0
    assert off.mean_cost == pytest.approx((4 * 1.0 + 2 * 0.5 + 2 * 10.0) / 4)
    m2_only = measure_cascade(s1, s2, None, 0.5, stage_costs=(1.0, 2.0), escalate_costs=esc)
    assert m2_only.mean_cost == pytest.approx((4 * 2.0 + 2 * 0.5 + 2 * 10.0) / 4)


def test_cost_objective_prefers_cheaper_point():
    s1, s2 = _stages(np.random.default_rng(72_000), 2000)
    g1, g2 = _grids()
    esc = np.full(2000, 1.0)
    # M2 is as expensive as escalating, so the cheapest certified point skips it.
    cert = certify_cascade(
        s1, s2, 0.05, lambda1_grid=g1, lambda2_grid=g2, confidence_level=0.9,
        stage_costs=(0.0, 1.0), escalate_costs=esc,
    )
    assert cert.feasible
    assert cert.objective == "cost"
    assert cert.lambda2 is None


def test_infeasible_below_floor():
    s1, s2 = _stages(np.random.default_rng(73_000), 80)
    g1, g2 = _grids()
    cert = certify_cascade(s1, s2, 0.05, lambda1_grid=g1, lambda2_grid=g2, confidence_level=0.9)
    assert not cert.feasible
    assert cert.min_n_required == 95
    assert any("Need at least 95" in note for note in cert.notes)


def test_rejects_non_binary_loss_and_mismatched_stages():
    s1, s2 = _stages(np.random.default_rng(74_000), 200)
    weighted = StageArrays(s1.confidence, s1.wrong * 3.0, s1.auto_cost)
    with pytest.raises(ValueError, match="0-1 loss"):
        certify_cascade(weighted, s2, 0.05, lambda1_grid=[0.8], lambda2_grid=[0.9])
    short = StageArrays(s2.confidence[:10], s2.wrong[:10], s2.auto_cost[:10])
    with pytest.raises(ValueError, match="same calibration cases"):
        certify_cascade(s1, short, 0.05, lambda1_grid=[0.8], lambda2_grid=[0.9])


def test_quantile_grid_is_label_free_and_falls_back():
    grid = quantile_grid([0.5, 0.6, 0.7, 0.8, 0.9, float("-inf")], 3)
    assert grid == sorted(grid) and len(grid) == 3
    assert all(0.5 < g < 0.9 for g in grid)
    assert quantile_grid([], 3) == pytest.approx([0.25, 0.5, 0.75])
