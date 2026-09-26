import pytest

from decisionops.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from decisionops.schemas import DecisionCase, DecisionPrediction


def _pair(label: str, prediction: str, probs: dict[str, float], cid: str = "x"):
    case = DecisionCase(id=cid, choices=list(probs.keys()), label=label)
    pred = DecisionPrediction(id=cid, prediction=prediction, probabilities=probs)
    return case, pred


def test_perfect_accuracy_and_low_nll():
    pairs = [
        _pair("a", "a", {"a": 0.9, "b": 0.1}, "1"),
        _pair("b", "b", {"a": 0.2, "b": 0.8}, "2"),
    ]
    assert accuracy(pairs) == 1.0
    assert negative_log_likelihood(pairs) < 0.3
    assert brier_score(pairs) < 0.1


def test_ece_zero_when_perfectly_calibrated_single_bin():
    # All conf=0.7 and accuracy=0.7 — construct 10 samples
    pairs = []
    for i in range(10):
        label = "a" if i < 7 else "b"
        # model always predicts a with conf 0.7
        pairs.append(_pair(label, "a", {"a": 0.7, "b": 0.3}, str(i)))
    # accuracy of prediction "a" is 0.7, conf always 0.7 => ECE ~ 0
    assert expected_calibration_error(pairs, n_bins=10) == pytest.approx(0.0, abs=1e-9)
