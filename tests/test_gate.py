"""Tests for temperature scaling, isotonic, and cal/test gate protocol."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from decisionops.calibration import (
    apply_isotonic,
    apply_temperature,
    fit_isotonic,
    fit_temperature,
)
from decisionops.eval import run_gate
from decisionops.io import load_dataset, load_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction
from decisionops.split import split_pairs

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "examples" / "gate-synthetic"
TOY = ROOT / "examples" / "toy-routing"


def _overconfident_pairs(n: int = 40):
    """Construct pairs where model is accurate ~70% but conf ~0.95."""
    pairs = []
    for i in range(n):
        correct = i % 10 < 7
        label = "a"
        pred_label = "a" if correct else "b"
        # peaky probs
        probs = {"a": 0.95 if pred_label == "a" else 0.05, "b": 0.05 if pred_label == "a" else 0.95}
        case = DecisionCase(id=str(i), choices=["a", "b"], label=label)
        pred = DecisionPrediction(id=str(i), prediction=pred_label, probabilities=probs)
        pairs.append((case, pred))
    return pairs


def test_temperature_scaling_increases_t_when_overconfident():
    pairs = _overconfident_pairs()
    ts = fit_temperature(pairs)
    # Overconfident => T should be > 1 to soften.
    assert ts.temperature > 1.0
    assert ts.nll_after <= ts.nll_before + 1e-6


def test_apply_temperature_preserves_argmax_for_positive_t():
    pred = DecisionPrediction(
        id="1",
        prediction="a",
        probabilities={"a": 0.8, "b": 0.15, "c": 0.05},
    )
    out = apply_temperature(pred, temperature=2.0)
    assert out.prediction == "a"
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-8
    assert out.confidence < pred.confidence  # softened


def test_isotonic_softens_overconfident_pairs():
    pairs = _overconfident_pairs()
    iso = fit_isotonic(pairs)
    assert iso.n_calibration == len(pairs)
    assert iso.nll_after <= iso.nll_before + 1e-6
    # Overconfident peaks (~0.95) should map down toward empirical accuracy (~0.7).
    calibrated = apply_isotonic(pairs[0][1], iso)
    assert calibrated.prediction == pairs[0][1].prediction
    assert calibrated.confidence < pairs[0][1].confidence
    assert abs(sum(calibrated.probabilities.values()) - 1.0) < 1e-8


def test_apply_isotonic_preserves_argmax():
    pairs = _overconfident_pairs(20)
    iso = fit_isotonic(pairs)
    pred = DecisionPrediction(
        id="x",
        prediction="a",
        probabilities={"a": 0.8, "b": 0.15, "c": 0.05},
    )
    out = apply_isotonic(pred, iso)
    assert out.prediction == "a"
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-8
    assert out.probabilities["a"] >= max(
        v for k, v in out.probabilities.items() if k != "a"
    )


def test_split_sizes():
    pairs = _overconfident_pairs(20)
    split = split_pairs(pairs, cal_ratio=0.3, seed=0)
    assert split.n_calibration + split.n_test == 20
    assert split.n_calibration >= 2
    assert split.n_test >= 2


def test_gate_on_toy_runs():
    cases = load_dataset(TOY / "dataset.jsonl")
    preds = load_predictions(TOY / "predictions.jsonl")
    report = run_gate(cases, preds, cal_ratio=0.3, seed=42, calibrate=True)
    assert report.split.n_total == len(cases)
    assert len(report.conditions) == 3
    assert [c.name for c in report.conditions] == [
        "raw",
        "temperature_scaled",
        "isotonic",
    ]
    assert report.temperature is not None and report.temperature > 0
    assert report.isotonic_fit.get("n_calibration", 0) > 0
    assert report.verdict.recommendation in {"go", "watch", "no-go"}
    # Thresholds on test should use cal-chosen threshold values.
    for cond in report.conditions:
        assert len(cond.thresholds) == 3
        assert len(cond.cal_thresholds) == 3
        for t, c in zip(cond.thresholds, cond.cal_thresholds, strict=True):
            assert t.threshold == c.threshold
            assert t.n_total == report.split.n_test


@pytest.mark.skipif(not GATE.exists(), reason="gate-synthetic example missing")
def test_gate_on_synthetic_2000():
    cases = load_dataset(GATE / "dataset.jsonl")
    preds = load_predictions(GATE / "predictions.jsonl")
    assert len(cases) >= 1000
    report = run_gate(cases, preds, cal_ratio=0.3, seed=42, calibrate=True)
    assert report.split.n_test >= 500
    names = {c.name for c in report.conditions}
    assert names == {"raw", "temperature_scaled", "isotonic"}
    for cond in report.conditions:
        assert 0.0 <= cond.accuracy <= 1.0
        assert cond.ece >= 0.0
        assert math.isfinite(cond.ece)
    # Temperature fit should be finite
    assert math.isfinite(report.temperature or float("nan"))
    assert report.isotonic_fit.get("n_calibration", 0) > 0


def _pool(n: int, *, conf_good: float, conf_bad: float, err_every: int = 10):
    """Binary pool: every `err_every`-th case wrong, at `conf_bad`; rest right at `conf_good`."""
    cases, preds = [], []
    for i in range(n):
        correct = i % err_every != 0
        conf = conf_good if correct else conf_bad
        cases.append(DecisionCase(id=str(i), choices=["a", "b"], label="a" if correct else "b"))
        preds.append(
            DecisionPrediction(id=str(i), prediction="a", probabilities={"a": conf, "b": 1 - conf})
        )
    return cases, preds


def test_verdict_go_when_certified_coverage_clears_the_bar():
    # Separable: errors sit at 0.55, so a 5% certificate accepts the ~90% correct mass.
    cases, preds = _pool(600, conf_good=0.97, conf_bad=0.55)
    v = run_gate(cases, preds, calibrate=False).verdict
    assert v.recommendation == "go"
    assert v.certified_feasible and v.certified_coverage >= 0.30
    assert v.verdict_alpha == 0.05 and v.min_coverage == 0.30


def test_verdict_watch_when_certified_but_below_min_coverage():
    cases, preds = _pool(600, conf_good=0.97, conf_bad=0.55)
    v = run_gate(cases, preds, calibrate=False, min_coverage=0.99).verdict
    assert v.certified_feasible
    assert v.recommendation == "watch"


def test_verdict_no_go_when_nothing_certifies():
    # Errors carry the same confidence as correct cases and exceed 5%: no threshold holds.
    cases, preds = _pool(600, conf_good=0.97, conf_bad=0.97, err_every=5)
    v = run_gate(cases, preds, calibrate=False).verdict
    assert not v.certified_feasible
    assert v.recommendation == "no-go"
    assert v.certified_coverage is None


def test_verdict_no_go_on_pool_too_small_to_split():
    cases, preds = _pool(8, conf_good=0.97, conf_bad=0.55)
    v = run_gate(cases, preds).verdict
    assert v.recommendation == "no-go"


def test_ece_does_not_move_the_verdict():
    """
    Regression for the old rule, which required an ECE drop for 'go'.

    A pool that is already calibrated (no ECE to gain) but certifies most traffic
    must be 'go'; one where calibration slashes ECE but nothing certifies must be
    'no-go'. On BANKING77 the old rule gave Jev (67% certified at 5%) and Laya
    (infeasible) the same 'watch'.
    """
    cases, preds = _pool(600, conf_good=0.97, conf_bad=0.55)
    good = run_gate(cases, preds, calibrate=True).verdict
    assert good.recommendation == "go"

    bad_cases, bad_preds = _pool(600, conf_good=0.99, conf_bad=0.99, err_every=3)
    bad = run_gate(bad_cases, bad_preds, calibrate=True).verdict
    assert bad.ece_relative_drop is not None and bad.ece_relative_drop > 0.15
    assert bad.recommendation == "no-go"


def test_gate_and_certify_agree_on_the_certificate():
    from decisionops.conformal import run_certification

    cases, preds = _pool(600, conf_good=0.97, conf_bad=0.55)
    v = run_gate(cases, preds, seed=3).verdict
    level = run_certification(cases, preds, alphas=[0.05], seed=3).levels[0]
    assert v.certified_threshold == level.certificate.threshold
    assert v.certified_coverage == pytest.approx(float(level.test["coverage"]))


def test_gate_rejects_bad_verdict_parameters():
    cases, preds = _pool(100, conf_good=0.97, conf_bad=0.55)
    with pytest.raises(ValueError, match="verdict_alpha"):
        run_gate(cases, preds, verdict_alpha=0.0)
    with pytest.raises(ValueError, match="min_coverage"):
        run_gate(cases, preds, min_coverage=1.5)
