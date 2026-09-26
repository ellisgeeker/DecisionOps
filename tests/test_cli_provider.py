"""CLI prediction-source resolution: --predictions vs --provider."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from decisionops.cli.main import app

runner = CliRunner()


def _write_pair(tmp_path, n: int = 40):
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ds, pred = [], []
    for i in range(n):
        ds.append(json.dumps({"id": str(i), "choices": ["a", "b"], "label": "a"}))
        pred.append(
            json.dumps(
                {
                    "id": str(i),
                    "prediction": "a",
                    "probabilities": {"a": 0.9, "b": 0.1},
                }
            )
        )
    dataset.write_text("\n".join(ds) + "\n", encoding="utf-8")
    predictions.write_text("\n".join(pred) + "\n", encoding="utf-8")
    return dataset, predictions


def test_eval_accepts_predictions_without_provider_flag(tmp_path):
    dataset, predictions = _write_pair(tmp_path)
    result = runner.invoke(app, ["eval", str(dataset), "-p", str(predictions)])
    assert result.exit_code == 0, result.output
    assert "DecisionOps Eval" in result.output


def test_eval_rejects_missing_source(tmp_path):
    dataset, _ = _write_pair(tmp_path)
    result = runner.invoke(app, ["eval", str(dataset)])
    assert result.exit_code != 0
    assert "--predictions" in result.output or "provider" in result.output.lower()


def test_eval_rejects_unknown_provider(tmp_path):
    dataset, predictions = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["eval", str(dataset), "-p", str(predictions), "--provider", "jev"]
    )
    assert result.exit_code != 0
    assert "unknown provider" in result.output.lower() or "jev" in result.output.lower()


def test_eval_rejects_predictions_with_laya(tmp_path):
    dataset, predictions = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["eval", str(dataset), "-p", str(predictions), "--provider", "laya"]
    )
    assert result.exit_code != 0
    assert "static" in result.output.lower()


def test_eval_limit_truncates(tmp_path):
    dataset, predictions = _write_pair(tmp_path, n=40)
    result = runner.invoke(
        app, ["eval", str(dataset), "-p", str(predictions), "--limit", "10"]
    )
    assert result.exit_code == 0, result.output
    assert "samples=10" in result.output


def test_gate_still_runs_with_dash_p(tmp_path):
    dataset, predictions = _write_pair(tmp_path, n=40)
    result = runner.invoke(app, ["gate", str(dataset), "-p", str(predictions)])
    assert result.exit_code == 0, result.output
    assert "Technical Gate" in result.output
