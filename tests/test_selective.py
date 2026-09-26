from decisionops.schemas import DecisionCase, DecisionPrediction
from decisionops.selective.threshold import evaluate_threshold, recommend_thresholds


def _pairs():
    rows = []
    # 5 correct high-conf, 5 wrong low-conf-ish
    for i in range(5):
        rows.append(
            (
                DecisionCase(id=f"c{i}", choices=["a", "b"], label="a"),
                DecisionPrediction(
                    id=f"c{i}",
                    prediction="a",
                    probabilities={"a": 0.95, "b": 0.05},
                ),
            )
        )
    for i in range(5, 10):
        rows.append(
            (
                DecisionCase(id=f"c{i}", choices=["a", "b"], label="a"),
                DecisionPrediction(
                    id=f"c{i}",
                    prediction="b",
                    probabilities={"a": 0.40, "b": 0.60},
                ),
            )
        )
    return rows


def test_high_threshold_accepts_only_confident():
    point = evaluate_threshold(_pairs(), threshold=0.9)
    assert point.n_accepted == 5
    assert point.coverage == 0.5
    assert point.selective_risk == 0.0


def test_recommend_target_risk():
    recs = recommend_thresholds(_pairs(), target_risks=[0.01, 0.5])
    assert len(recs) == 2
    # at 1% risk, only the perfect high-conf bucket works
    assert recs[0].selective_risk <= 0.01 + 1e-9
    assert recs[0].coverage == 0.5
