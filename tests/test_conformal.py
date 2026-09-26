"""Tests for Conformal Risk Control."""

from __future__ import annotations

import math

import pytest

from decisionops.conformal import certify_threshold, min_calibration_samples
from decisionops.conformal.certify import run_certification
from decisionops.conformal.risk_control import evaluate_certified_threshold
from decisionops.schemas import DecisionCase, DecisionPrediction


def _pair(cid: str, conf: float, correct: bool):
    """
    Binary case whose max probability is `conf`; `correct` sets the label.

    Binary forces max prob >= 0.5, so `conf` below that is not representable —
    guard against it rather than silently building a different example.
    """
    if not 0.5 <= conf <= 1.0:
        raise ValueError(f"binary confidence must be in [0.5, 1.0], got {conf}")
    pred_label = "a"
    label = "a" if correct else "b"
    case = DecisionCase(id=cid, choices=["a", "b"], label=label)
    pred = DecisionPrediction(
        id=cid,
        prediction=pred_label,
        probabilities={"a": conf, "b": 1.0 - conf},
    )
    return case, pred


def _pair3(cid: str, conf: float, correct: bool):
    """Three-way case, so confidences below 0.5 are representable."""
    rest = (1.0 - conf) / 2.0
    pred_label = "a"
    label = "a" if correct else "b"
    case = DecisionCase(id=cid, choices=["a", "b", "c"], label=label)
    pred = DecisionPrediction(
        id=cid,
        prediction=pred_label,
        probabilities={"a": conf, "b": rest, "c": rest},
    )
    return case, pred


def test_min_calibration_samples_matches_bound_algebra():
    # alpha >= B/(n+1)  =>  n >= B/alpha - 1
    assert min_calibration_samples(0.01) == 99
    assert min_calibration_samples(0.05) == 19
    assert min_calibration_samples(0.10) == 9
    with pytest.raises(ValueError):
        min_calibration_samples(0.0)


def test_infeasible_when_calibration_set_too_small():
    # 20 samples cannot certify alpha=1%: B/(n+1) = 1/21 > 0.01
    pairs = [_pair(str(i), 0.99, True) for i in range(20)]
    result = certify_threshold(pairs, alpha=0.01)
    assert result.feasible is False
    assert result.threshold is None
    assert result.min_n_required == 99
    assert any("Infeasible" in n for n in result.notes)


def test_certificate_satisfies_the_crc_bound():
    # 200 samples: high-confidence correct, low-confidence wrong.
    pairs = [_pair(f"g{i}", 0.99, True) for i in range(150)]
    pairs += [_pair(f"b{i}", 0.60, False) for i in range(50)]
    result = certify_threshold(pairs, alpha=0.05)

    assert result.feasible is True
    assert result.threshold is not None
    n = result.n_calibration
    # Recompute the bound independently.
    expected = (n / (n + 1)) * result.empirical_joint_error + 1.0 / (n + 1)
    assert result.crc_bound_value == pytest.approx(expected)
    assert result.crc_bound_value <= 0.05 + 1e-12
    # Errors sit at 0.60, so the threshold must exclude them.
    assert result.threshold > 0.60
    assert result.coverage_calibration == pytest.approx(150 / 200)


def test_threshold_is_smallest_feasible_so_coverage_is_maximal():
    pairs = [_pair(f"g{i}", 0.90, True) for i in range(120)]
    pairs += [_pair(f"h{i}", 0.95, True) for i in range(80)]
    result = certify_threshold(pairs, alpha=0.05)
    # No errors anywhere, so the loosest observed threshold should win.
    assert result.threshold == pytest.approx(0.90)
    assert result.coverage_calibration == pytest.approx(1.0)


def test_stricter_alpha_never_lowers_the_threshold():
    pairs = [_pair(f"g{i}", 0.99, True) for i in range(300)]
    pairs += [_pair(f"m{i}", 0.80, i % 2 == 0) for i in range(100)]
    loose = certify_threshold(pairs, alpha=0.10)
    strict = certify_threshold(pairs, alpha=0.02)
    assert loose.feasible and strict.feasible
    assert strict.threshold >= loose.threshold


def test_evaluate_certified_threshold_reports_joint_and_selective():
    # Third case has max prob 0.30, so the threshold abstains on it.
    pairs = [_pair("a", 0.99, True), _pair("b", 0.99, False), _pair3("c", 0.30, False)]
    stats = evaluate_certified_threshold(pairs, threshold=0.5)
    assert stats["n_accepted"] == 2
    assert stats["coverage"] == pytest.approx(2 / 3)
    # 1 wrong out of 3 total
    assert stats["joint_error_rate"] == pytest.approx(1 / 3)
    # 1 wrong out of 2 accepted
    assert stats["selective_risk"] == pytest.approx(0.5)


def test_pair_helper_rejects_unrepresentable_binary_confidence():
    with pytest.raises(ValueError):
        _pair("x", 0.10, False)


def test_run_certification_end_to_end():
    cases, preds = [], []
    for i in range(400):
        correct = i % 10 != 0
        conf = 0.97 if correct else 0.55
        case, pred = _pair(str(i), conf, correct)
        cases.append(case)
        preds.append(pred)

    report = run_certification(
        cases, preds, alphas=[0.05, 0.10], cal_ratio=0.3, seed=1, calibrate=False
    )
    assert report.split.n_calibration + report.split.n_test == 400
    assert len(report.levels) == 2
    for level in report.levels:
        if level.certificate.feasible:
            assert level.test is not None
            assert math.isfinite(float(level.test["joint_error_rate"]))
            assert level.bound_held_on_test is not None


def _calibrated_pool(rng, n: int):
    """Perfectly calibrated synthetic pool: P(correct) == reported confidence."""
    cases, preds = [], []
    for i in range(n):
        conf = float(rng.uniform(0.55, 0.999))
        correct = bool(rng.random() < conf)
        case, pred = _pair(str(i), conf, correct)
        cases.append(case)
        preds.append(pred)
    return cases, preds


def test_guarantee_holds_in_expectation_over_fresh_samples():
    """
    CRC bounds E[L(lambda_hat)] over draws of the whole exchangeable sample.

    Two traps this test avoids:

    1. Reshuffling one fixed pool is NOT resampling. Every split inherits that
       pool's sampling error, so the average estimates the pool's own error rate,
       not the distributional expectation. Each repetition draws a fresh pool.
    2. `mean <= alpha` is a knife-edge assertion. CRC picks the smallest feasible
       lambda, so true risk sits just below alpha and a finite average straddles
       it. The comparison uses a standard-error tolerance instead.
    """
    import numpy as np

    alpha, reps = 0.10, 25
    joint_errors = []
    for rep in range(reps):
        cases, preds = _calibrated_pool(np.random.default_rng(10_000 + rep), 800)
        report = run_certification(
            cases, preds, alphas=[alpha], cal_ratio=0.4, seed=rep, calibrate=False
        )
        level = report.levels[0]
        if not level.certificate.feasible:
            continue
        joint_errors.append(float(level.test["joint_error_rate"]))

    assert len(joint_errors) >= 20
    errors = np.asarray(joint_errors)
    se = float(errors.std(ddof=1) / np.sqrt(len(errors)))
    # Allow 3 standard errors: the bound is on the expectation, estimated here.
    assert errors.mean() <= alpha + 3.0 * se, (
        f"mean joint error {errors.mean():.5f} exceeds alpha={alpha} "
        f"by more than 3 se ({se:.5f}) - guarantee likely violated"
    )


def test_per_split_outcomes_are_expected_to_straddle_alpha():
    """Documents why no single-split assertion belongs in this suite."""
    import numpy as np

    alpha = 0.10
    held = []
    for rep in range(20):
        cases, preds = _calibrated_pool(np.random.default_rng(500 + rep), 800)
        report = run_certification(
            cases, preds, alphas=[alpha], cal_ratio=0.4, seed=rep, calibrate=False
        )
        level = report.levels[0]
        if level.certificate.feasible:
            held.append(bool(level.bound_held_on_test))

    assert len(held) >= 15
    # Neither all-true nor all-false: a single split is not evidence either way.
    assert 0 < sum(held) < len(held)


def test_temperature_fit_uses_disjoint_samples():
    """Temperature must not be fit on the data CRC calibrates against."""
    cases, preds = [], []
    for i in range(500):
        correct = i % 8 != 0
        case, pred = _pair(str(i), 0.95 if correct else 0.60, correct)
        cases.append(case)
        preds.append(pred)

    report = run_certification(
        cases, preds, alphas=[0.10], cal_ratio=0.4, temp_ratio=0.25, seed=3
    )
    assert report.calibrated is True
    assert report.n_temperature_fit > 0
    # Temperature slice is carved out of the calibration pool, so the CRC set shrinks.
    assert report.split.n_calibration < 500 * 0.4
    assert report.n_temperature_fit + report.split.n_calibration <= 500 * 0.4 + 1
