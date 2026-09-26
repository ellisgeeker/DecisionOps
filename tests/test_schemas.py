import pytest
from pydantic import ValidationError

from decisionops.schemas import DecisionCase, DecisionPrediction


def test_case_label_must_be_in_choices():
    with pytest.raises(ValidationError):
        DecisionCase(
            id="1",
            choices=["a", "b"],
            label="c",
        )


def test_prediction_renormalizes_mild_drift():
    pred = DecisionPrediction(
        id="1",
        prediction="a",
        probabilities={"a": 0.6, "b": 0.45},  # sums to 1.05
    )
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-9
    assert pred.confidence == pytest.approx(max(pred.probabilities.values()))


def test_prediction_rejects_negative():
    with pytest.raises(ValidationError):
        DecisionPrediction(
            id="1",
            prediction="a",
            probabilities={"a": 0.9, "b": -0.1},
        )
