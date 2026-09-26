"""Tests for multi-model comparison under a shared gate protocol."""

from __future__ import annotations

import pytest

from decisionops.compare import compare_models, load_prediction_sets
from decisionops.schemas import DecisionCase, DecisionPrediction


def _cases(n: int = 60) -> list[DecisionCase]:
    return [
        DecisionCase(id=str(i), choices=["a", "b"], label="a" if i % 4 else "b")
        for i in range(n)
    ]


def _confident_good(cases: list[DecisionCase]) -> list[DecisionPrediction]:
    """Correct and sharp: should hold low risk with decent coverage."""
    out = []
    for c in cases:
        probs = {"a": 0.97, "b": 0.03} if c.label == "a" else {"a": 0.03, "b": 0.97}
        out.append(
            DecisionPrediction(
                id=c.id,
                prediction=c.label,
                probabilities=probs,
                latency_ms=10.0,
                provider="p_good",
                model="good",
            )
        )
    return out


def _confident_wrong(cases: list[DecisionCase]) -> list[DecisionPrediction]:
    """Frequently wrong but still sharp: overconfident, should rank worse."""
    out = []
    for i, c in enumerate(cases):
        wrong = i % 3 == 0
        pred = ("b" if c.label == "a" else "a") if wrong else c.label
        probs = {"a": 0.9, "b": 0.1} if pred == "a" else {"a": 0.1, "b": 0.9}
        out.append(
            DecisionPrediction(
                id=c.id,
                prediction=pred,
                probabilities=probs,
                latency_ms=50.0,
                provider="p_bad",
                model="bad",
            )
        )
    return out


def test_compare_uses_shared_split_and_ranks_by_coverage():
    # CRC at 5% needs n_cal >= 19; 120 * 0.4 leaves enough after the split.
    cases = _cases(120)
    report = compare_models(
        cases,
        {"good": _confident_good(cases), "bad": _confident_wrong(cases)},
        dataset_name="unit",
        cal_ratio=0.4,
        seed=7,
        target_risks=[0.05],
        calibrate=False,
    )
    assert report.dataset == "unit"
    assert len(report.entries) == 2
    # Shared split: both entries score the same number of test rows.
    assert len({e.n_test for e in report.entries}) == 1
    assert report.split.n_calibration + report.split.n_test == len(cases)
    assert report.metadata["protocol"] == "shared_split_gate_and_crc"
    assert report.metadata["certified_quantity"] == "joint_automated_error_rate"

    good = next(e for e in report.entries if e.name == "good")
    bad = next(e for e in report.entries if e.name == "bad")
    assert good.accuracy > bad.accuracy
    # Ranking is CRC-based; the accurate model should still win when both certify.
    assert report.best_by_coverage_at["0.05"] == "good"
    assert report.metadata["n_feasible_by_risk"]["0.05"] >= 1
    assert good.certified_feasible_by_risk["0.05"] is True


def test_infeasible_model_is_excluded_from_ranking():
    """A model with no CRC certificate must not win, even if empirical coverage is high."""
    # α=1% needs n_cal >= 99; 300 * 0.4 = 120.
    n = 300
    cases = [
        DecisionCase(id=str(i), choices=["a", "b"], label="a")
        for i in range(n)
    ]

    def sharp_correct() -> list[DecisionPrediction]:
        return [
            DecisionPrediction(
                id=c.id,
                prediction="a",
                probabilities={"a": 0.99, "b": 0.01},
                provider="good",
                model="good",
            )
            for c in cases
        ]

    def coin_flip_sharp() -> list[DecisionPrediction]:
        # ~50% accuracy, all mass at 0.99: no 1% certificate can exist.
        out = []
        for i, c in enumerate(cases):
            pred = "a" if i % 2 == 0 else "b"
            out.append(
                DecisionPrediction(
                    id=c.id,
                    prediction=pred,
                    probabilities={"a": 0.99, "b": 0.01} if pred == "a" else {"a": 0.01, "b": 0.99},
                    provider="coin",
                    model="coin",
                )
            )
        return out

    report = compare_models(
        cases,
        {"good": sharp_correct(), "coin": coin_flip_sharp()},
        cal_ratio=0.4,
        seed=1,
        target_risks=[0.01],
        calibrate=False,
    )
    coin = next(e for e in report.entries if e.name == "coin")
    assert coin.certified_feasible_by_risk["0.01"] is False
    assert "0.01" not in coin.certified_coverage_by_risk
    assert report.best_by_coverage_at.get("0.01") == "good"
    assert any("infeasible" in n for n in report.notes)


def test_test_split_overshoot_does_not_void_a_certificate():
    """
    bound_held_on_test is a coin flip; ranking must not use it.

    A feasible certificate whose test joint error landed above alpha still
    participates. That is the documented CRC semantics.
    """
    cases = _cases(400)
    report = compare_models(
        cases,
        {"good": _confident_good(cases)},
        cal_ratio=0.3,
        seed=0,
        target_risks=[0.10],
        calibrate=False,
    )
    entry = report.entries[0]
    # _risk_key(0.10) strips to "0.1".
    assert entry.certified_feasible_by_risk["0.1"] is True
    assert report.best_by_coverage_at["0.1"] == "good"
    # The empirical ! path is independent and must still be populated.
    assert "0.1" in entry.coverage_by_risk
    assert "0.1" in entry.selective_risk_by_risk


def test_compare_requires_predictions():
    with pytest.raises(ValueError):
        compare_models(_cases(), {})


def test_load_prediction_sets_rejects_duplicate_names(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text(
        '{"id":"0","prediction":"a","probabilities":{"a":0.9,"b":0.1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_prediction_sets([f"m={p}", f"m={p}"])


def test_load_prediction_sets_infers_name(tmp_path):
    p = tmp_path / "word_lr.jsonl"
    p.write_text(
        '{"id":"0","prediction":"a","probabilities":{"a":0.9,"b":0.1}}\n',
        encoding="utf-8",
    )
    sets = load_prediction_sets([str(p)])
    assert list(sets) == ["word_lr"]
