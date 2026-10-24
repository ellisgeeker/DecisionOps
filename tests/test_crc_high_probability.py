"""
High-probability certificates (`confidence_level`): RCPS with Clopper-Pearson.

CRC bounds E[joint error]. With `confidence_level = 1 - delta` the claim is
stronger: over the calibration draw, P(true joint error <= alpha) >= 1 - delta.
The checks below measure that directly against the closed-form true risk from
`tests/test_crc_analytic.py`, so they test the guarantee itself rather than a
noisy test split.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from decisionops.conformal import (
    certify_threshold,
    clopper_pearson_upper,
    min_calibration_samples_hp,
    run_certification,
)
from decisionops.schemas import DecisionCase, DecisionPrediction

LO, HI = 0.55, 0.999
SPAN = HI - LO


def true_joint_error(lam: float) -> float:
    """Closed-form joint error of accepting everything at or above `lam`."""
    lam = min(max(lam, LO), HI)
    return ((1.0 - lam) ** 2 - (1 - HI) ** 2) / (2.0 * SPAN)


def _pool(rng: np.random.Generator, n: int):
    """Binary pool where P(correct) == reported confidence exactly."""
    pairs = []
    for i in range(n):
        conf = float(rng.uniform(LO, HI))
        correct = bool(rng.random() < conf)
        pairs.append(
            (
                DecisionCase(id=str(i), choices=["a", "b"], label="a" if correct else "b"),
                DecisionPrediction(
                    id=str(i), prediction="a", probabilities={"a": conf, "b": 1.0 - conf}
                ),
            )
        )
    return pairs


def test_clopper_pearson_matches_binomial_tail():
    # Upper limit u satisfies P(Binom(n, u) <= k) = delta.
    for k, n in [(0, 50), (3, 200), (12, 400)]:
        u = clopper_pearson_upper(k, n, 0.95)
        assert binom.cdf(k, n, u) == pytest.approx(0.05, abs=1e-6)
        assert k / n < u < 1.0
    assert clopper_pearson_upper(5, 5, 0.95) == 1.0


def test_min_samples_hp_matches_zero_error_algebra():
    # n >= ln(delta) / ln(1 - alpha)
    assert min_calibration_samples_hp(0.01, 0.95) == 299
    assert min_calibration_samples_hp(0.05, 0.95) == 59
    n = min_calibration_samples_hp(0.05, 0.95)
    assert clopper_pearson_upper(0, n, 0.95) <= 0.05
    assert clopper_pearson_upper(0, n - 1, 0.95) > 0.05


def test_hp_is_stricter_than_crc_on_the_same_data():
    pairs = _pool(np.random.default_rng(7), 600)
    crc = certify_threshold(pairs, 0.05)
    hp = certify_threshold(pairs, 0.05, confidence_level=0.95)
    assert crc.guarantee == "expectation" and crc.confidence_level is None
    assert hp.guarantee == "high_probability" and hp.confidence_level == 0.95
    assert crc.feasible and hp.feasible
    assert hp.threshold >= crc.threshold
    assert hp.coverage_calibration <= crc.coverage_calibration
    assert hp.crc_bound_value <= 0.05


def test_hp_infeasible_when_zero_errors_cannot_clear_alpha():
    # 200 perfect samples clear CRC at 1% (needs 99) but not 95% HP (needs 299).
    pairs = [
        (
            DecisionCase(id=str(i), choices=["a", "b"], label="a"),
            DecisionPrediction(id=str(i), prediction="a", probabilities={"a": 0.99, "b": 0.01}),
        )
        for i in range(200)
    ]
    assert certify_threshold(pairs, 0.01).feasible
    hp = certify_threshold(pairs, 0.01, confidence_level=0.95)
    assert not hp.feasible
    assert hp.min_n_required == 299
    assert any("Infeasible" in note for note in hp.notes)


def test_hp_rejects_bad_arguments():
    pairs = _pool(np.random.default_rng(1), 50)
    with pytest.raises(ValueError, match="confidence_level"):
        certify_threshold(pairs, 0.05, confidence_level=1.0)
    with pytest.raises(ValueError, match="0-1 loss"):
        certify_threshold(pairs, 0.05, confidence_level=0.9, loss_bound=2.0)


def test_hp_violation_rate_is_at_most_delta():
    """
    Measure the guarantee: true risk at lambda_hat above alpha in <= delta of draws.

    delta=0.2 so the target rate is large enough to estimate with 120 reps; the
    tolerance is a binomial 99.9% upper band. Plain CRC on the same pools must
    violate far more often — otherwise this test would not distinguish them.
    """
    alpha, level, n_cal, reps = 0.05, 0.8, 300, 120
    delta = 1.0 - level
    hp_viol = crc_viol = feasible = 0
    for rep in range(reps):
        pairs = _pool(np.random.default_rng(40_000 + rep), n_cal)
        hp = certify_threshold(pairs, alpha, confidence_level=level)
        crc = certify_threshold(pairs, alpha)
        if hp.feasible:
            feasible += 1
            hp_viol += true_joint_error(hp.threshold) > alpha
        if crc.feasible:
            crc_viol += true_joint_error(crc.threshold) > alpha

    assert feasible >= 100
    allowed = binom.ppf(0.999, feasible, delta)
    assert hp_viol <= allowed, f"{hp_viol}/{feasible} violations > delta={delta}"
    # CRC only controls the mean, so roughly half its certificates sit above alpha.
    assert crc_viol > 2 * hp_viol
    assert crc_viol / reps > 0.25


def test_run_certification_records_guarantee():
    pairs = _pool(np.random.default_rng(3), 1200)
    cases = [c for c, _ in pairs]
    preds = [p for _, p in pairs]
    report = run_certification(
        cases, preds, alphas=[0.05], calibrate=False, confidence_level=0.9
    )
    assert report.metadata["guarantee"] == "high_probability"
    assert report.metadata["confidence_level"] == 0.9
    cert = report.levels[0].certificate
    assert cert.guarantee == "high_probability"
    assert any("High-probability guarantee" in n for n in report.notes)
    assert not any("E[loss]" in n for n in report.notes)
    assert math.isfinite(cert.crc_bound_value)
