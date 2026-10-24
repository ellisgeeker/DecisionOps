"""Tests for the `decisionops certify` command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from decisionops.cli.main import app

runner = CliRunner()


def _write_pool(tmp_path, n: int, *, conf_good: float = 0.97, conf_bad: float = 0.55):
    """
    Binary pool where 1 in 10 cases is wrong and carries low confidence.

    Separable by design, so a 5% or 10% certificate exists and a strict level can
    be checked for feasibility rather than correctness of the threshold value.
    """
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    ds_lines, pred_lines = [], []
    for i in range(n):
        correct = i % 10 != 0
        conf = conf_good if correct else conf_bad
        ds_lines.append(
            json.dumps({"id": str(i), "choices": ["a", "b"], "label": "a" if correct else "b"})
        )
        pred_lines.append(
            json.dumps(
                {
                    "id": str(i),
                    "prediction": "a",
                    "probabilities": {"a": conf, "b": 1.0 - conf},
                    "provider": "static",
                    "model": "fixture",
                }
            )
        )
    dataset.write_text("\n".join(ds_lines) + "\n", encoding="utf-8")
    predictions.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")
    return dataset, predictions


def test_certify_writes_report_and_prints_thresholds(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 400)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "certify",
            str(dataset),
            "--predictions",
            str(predictions),
            "--output",
            str(out),
            "--alphas",
            "0.05,0.10",
            "--no-calibrate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Conformal Risk Control" in result.output
    # The certified quantity must be named, not silently conflated with selective risk.
    assert "joint automated error" in result.output
    # "E[loss]" must survive rendering: rich eats bracketed text as markup unless escaped.
    assert "E[loss]" in result.output

    path = out / "certification_report.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [lv["certificate"]["alpha"] for lv in payload["levels"]] == [0.05, 0.10]
    assert payload["metadata"]["protocol"] == "three_way_split_temp_crc_test"
    for level in payload["levels"]:
        if level["certificate"]["feasible"]:
            assert level["test"] is not None
            assert level["bound_held_on_test"] is not None


def test_certify_surfaces_infeasible_levels(tmp_path):
    # 60 rows leaves ~42 for CRC after the split: B/(n+1) > 0.01, so 1% cannot hold.
    dataset, predictions = _write_pool(tmp_path, 60)

    result = runner.invoke(
        app,
        [
            "certify",
            str(dataset),
            "--predictions",
            str(predictions),
            "--alphas",
            "0.01",
            "--no-calibrate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "INFEASIBLE" in result.output
    assert "No certificate at 1%" in result.output


def test_certify_applies_temperature_on_a_disjoint_slice(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 500)

    result = runner.invoke(
        app,
        [
            "certify",
            str(dataset),
            "--predictions",
            str(predictions),
            "--alphas",
            "0.10",
            "--temp-ratio",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "temperature T=" in result.output
    assert "disjoint from the CRC set" in result.output


def test_certify_with_contract_labels_miss_rate(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 400)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"name": "fixture", "choices": ["a", "b"], "auto_choices": ["a"]}),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "certify", str(dataset), "-p", str(predictions),
            "--contract", str(contract), "--alphas", "0.10",
            "--no-calibrate", "-o", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "automated miss rate" in result.output
    payload = json.loads((out / "certification_report.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["contract"]["auto_choices"] == ["a"]


def test_certify_rejects_contract_that_does_not_fit_data(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 120)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"name": "wrong", "choices": ["x", "y"], "auto_choices": ["x"]}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["certify", str(dataset), "-p", str(predictions), "--contract", str(contract)]
    )
    assert result.exit_code != 0
    assert "does not fit contract" in result.output


def test_certify_confidence_level_gives_high_probability_certificate(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 800)
    out = tmp_path / "out"
    profile = tmp_path / "profile.json"

    result = runner.invoke(
        app,
        [
            "certify", str(dataset), "-p", str(predictions),
            "--alphas", "0.10", "--no-calibrate", "--confidence-level", "0.95",
            "-o", str(out), "--profile", str(profile),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "high-probability" in result.output
    assert "Upper bound" in result.output
    payload = json.loads((out / "certification_report.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["guarantee"] == "high_probability"
    assert payload["metadata"]["confidence_level"] == 0.95
    cert = payload["levels"][0]["certificate"]
    assert cert["feasible"] and cert["guarantee"] == "high_probability"
    frozen = json.loads(profile.read_text(encoding="utf-8"))
    assert frozen["guarantee"] == "high_probability"
    assert frozen["confidence_level"] == 0.95


def test_certify_default_guarantee_is_expectation(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 400)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["certify", str(dataset), "-p", str(predictions), "--alphas", "0.10",
         "--no-calibrate", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "in expectation" in result.output
    payload = json.loads((out / "certification_report.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["guarantee"] == "expectation"
    assert payload["metadata"]["confidence_level"] is None


def test_certify_rejects_out_of_range_confidence_level(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 120)
    for value in ("0", "1", "1.5"):
        result = runner.invoke(
            app,
            ["certify", str(dataset), "-p", str(predictions), "--confidence-level", value],
        )
        assert result.exit_code != 0, value


def test_certify_rejects_out_of_range_ratios(tmp_path):
    dataset, predictions = _write_pool(tmp_path, 120)
    for flag, value in [("--cal-ratio", "1.0"), ("--temp-ratio", "1.0"), ("--alphas", "1.5")]:
        result = runner.invoke(
            app,
            ["certify", str(dataset), "--predictions", str(predictions), flag, value],
        )
        assert result.exit_code != 0, f"{flag}={value} should have been rejected"
