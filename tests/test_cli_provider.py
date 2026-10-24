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
        app, ["eval", str(dataset), "-p", str(predictions), "--provider", "nope"]
    )
    assert result.exit_code != 0
    assert "unknown provider" in result.output.lower()


def test_eval_rejects_predictions_with_jev(tmp_path):
    dataset, predictions = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["eval", str(dataset), "-p", str(predictions), "--provider", "jev"]
    )
    assert result.exit_code != 0
    assert "static" in result.output.lower()


def test_provider_specific_flags_rejected_on_wrong_provider(tmp_path):
    dataset, predictions = _write_pair(tmp_path)
    cases = [
        ["--no-builtin-temperature"],
        ["--jev-base-url", "https://x.example"],
        ["--jev-model", "ft-1"],
        ["--jev-question", "noul"],
        ["--http-url", "https://model.example/v1/predict"],
        ["--http-token", "sekrit"],
    ]
    for extra in cases:
        result = runner.invoke(app, ["eval", str(dataset), "-p", str(predictions), *extra])
        assert result.exit_code != 0, extra
        assert "not valid with --provider static" in result.output, extra


def test_eval_runs_jev_with_contract_descriptions(tmp_path, monkeypatch):
    """--provider jev end to end, with the provider stubbed so no network is used."""
    import decisionops.adapters.jev as jev_mod
    from decisionops.schemas import DecisionPrediction

    seen = {}

    class StubJev:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.model = kwargs["model"]

        def predict_many(self, cases):
            return [
                DecisionPrediction(
                    id=c.id, prediction="a", probabilities={"a": 0.9, "b": 0.1}, provider="jev"
                )
                for c in cases
            ]

    monkeypatch.setattr(jev_mod, "JevProvider", StubJev)
    dataset, _ = _write_pair(tmp_path)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "name": "t",
                "choices": ["a", "b"],
                "auto_choices": ["a"],
                "descriptions": {"a": "safe"},
                "instructions": "Pick.",
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "eval", str(dataset), "--provider", "jev",
            "--jev-model", "ft-1", "--jev-base-url", "https://jev.internal.example",
            "--contract", str(contract),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["model"] == "ft-1"
    assert seen["base_url"] == "https://jev.internal.example"
    assert seen["descriptions"] == {"a": "safe"}
    assert seen["contract_instructions"] == "Pick."
    assert seen["question_type"] == "choice"


def test_eval_runs_http_with_contract_descriptions(tmp_path, monkeypatch):
    """--provider http end to end, opener-free: the class is stubbed."""
    import decisionops.adapters.http as http_mod
    from decisionops.schemas import DecisionPrediction

    seen = {}

    class StubHttp:
        def __init__(self, url, **kwargs):
            seen["url"] = url
            seen.update(kwargs)
            self.url = url

        def predict_many(self, cases):
            return [
                DecisionPrediction(
                    id=c.id, prediction="a", probabilities={"a": 0.9, "b": 0.1}, provider="http"
                )
                for c in cases
            ]

    monkeypatch.setattr(http_mod, "HttpProvider", StubHttp)
    dataset, _ = _write_pair(tmp_path)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "name": "t",
                "choices": ["a", "b"],
                "auto_choices": ["a"],
                "descriptions": {"a": "safe"},
                "instructions": "Pick.",
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "eval", str(dataset), "--provider", "http",
            "--http-url", "https://model.example/v1/predict",
            "--http-token", "sekrit",
            "--contract", str(contract),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["url"] == "https://model.example/v1/predict"
    assert seen["token"] == "sekrit"
    assert seen["descriptions"] == {"a": "safe"}
    assert seen["contract_instructions"] == "Pick."


def test_http_requires_url(tmp_path):
    dataset, _ = _write_pair(tmp_path)
    result = runner.invoke(app, ["eval", str(dataset), "--provider", "http"])
    assert result.exit_code != 0
    assert "--http-url" in result.output


def test_eval_rejects_predictions_with_http(tmp_path):
    dataset, predictions = _write_pair(tmp_path)
    result = runner.invoke(
        app,
        [
            "eval", str(dataset), "-p", str(predictions),
            "--provider", "http", "--http-url", "https://model.example/v1/predict",
        ],
    )
    assert result.exit_code != 0
    assert "static" in result.output.lower()


def test_jev_noul_requires_contract_with_yes_choice(tmp_path):
    dataset, _ = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["eval", str(dataset), "--provider", "jev", "--jev-question", "noul"]
    )
    assert result.exit_code != 0
    assert "yes_choice" in result.output


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
    assert "Basis: CRC certificate at 5%" in result.output


def test_gate_verdict_flags(tmp_path):
    dataset, predictions = _write_pair(tmp_path, n=400)
    result = runner.invoke(
        app,
        [
            "gate", str(dataset), "-p", str(predictions),
            "--verdict-alpha", "0.10", "--min-coverage", "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CRC certificate at 10%" in result.output
    assert ">= 50%" in result.output
    for flag, value in [("--verdict-alpha", "0"), ("--min-coverage", "1.5")]:
        bad = runner.invoke(app, ["gate", str(dataset), "-p", str(predictions), flag, value])
        assert bad.exit_code != 0, flag
