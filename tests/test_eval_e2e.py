from pathlib import Path

from decisionops.adapters import StaticFileProvider
from decisionops.eval import evaluate
from decisionops.io import load_dataset

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy-routing"


def test_toy_routing_eval_runs():
    cases = load_dataset(TOY / "dataset.jsonl")
    preds = StaticFileProvider(str(TOY / "predictions.jsonl")).predict_many(cases)
    result = evaluate(cases, preds, target_risks=[0.01, 0.05, 0.10])
    assert result.n_samples == len(cases)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.ece >= 0.0
    assert result.brier >= 0.0
    assert result.nll >= 0.0
    assert len(result.thresholds) == 3
    assert result.latency.count == len(cases)
