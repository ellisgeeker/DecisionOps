"""Tests for the `decisionops compare` command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from decisionops.cli.main import app

runner = CliRunner()


def _write_two_models(tmp_path, n: int = 200):
    """
    One dataset, two prediction files with different confidence profiles.

    `sharp` is overconfident (high peak when right and wrong); `soft` is flatter.
    Separable enough that CRC can issue a certificate at 10% on the larger pool.
    """
    dataset = tmp_path / "dataset.jsonl"
    sharp = tmp_path / "sharp.jsonl"
    soft = tmp_path / "soft.jsonl"
    ds, sharp_lines, soft_lines = [], [], []
    for i in range(n):
        correct = i % 10 != 0
        label = "a" if correct else "b"
        ds.append(json.dumps({"id": str(i), "choices": ["a", "b"], "label": label}))
        sharp_conf = 0.97 if correct else 0.80
        soft_conf = 0.70 if correct else 0.55
        sharp_lines.append(
            json.dumps(
                {
                    "id": str(i),
                    "prediction": "a",
                    "probabilities": {"a": sharp_conf, "b": 1.0 - sharp_conf},
                    "provider": "static",
                    "model": "sharp",
                }
            )
        )
        soft_lines.append(
            json.dumps(
                {
                    "id": str(i),
                    "prediction": "a",
                    "probabilities": {"a": soft_conf, "b": 1.0 - soft_conf},
                    "provider": "static",
                    "model": "soft",
                }
            )
        )
    dataset.write_text("\n".join(ds) + "\n", encoding="utf-8")
    sharp.write_text("\n".join(sharp_lines) + "\n", encoding="utf-8")
    soft.write_text("\n".join(soft_lines) + "\n", encoding="utf-8")
    return dataset, sharp, soft


def test_compare_writes_report_and_prints_both_tables(tmp_path):
    dataset, sharp, soft = _write_two_models(tmp_path, n=240)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "compare",
            str(dataset),
            "-p",
            f"sharp={sharp}",
            "-p",
            f"soft={soft}",
            "--output",
            str(out),
            "--target-risks",
            "0.05,0.10",
            "--no-calibrate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Model Comparison" in result.output
    # Both ranking surfaces must appear: CRC is primary, empirical is contrast.
    assert "Certified coverage" in result.output
    assert "Empirical coverage" in result.output
    assert "joint automated error rate" in result.output
    assert "sharp" in result.output
    assert "soft" in result.output

    path = out / "compare_report.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {e["name"] for e in payload["entries"]}
    assert names == {"sharp", "soft"}
    assert payload["target_risks"] == [0.05, 0.10]
    assert (out / "compare_report.html").exists()


def test_compare_rejects_single_prediction_set(tmp_path):
    dataset, sharp, _ = _write_two_models(tmp_path, n=80)

    result = runner.invoke(
        app,
        ["compare", str(dataset), "-p", f"sharp={sharp}"],
    )

    assert result.exit_code != 0
    assert "at least two" in result.output.lower()


def test_compare_rejects_bad_cal_ratio(tmp_path, plain_output):
    dataset, sharp, soft = _write_two_models(tmp_path, n=80)

    result = runner.invoke(
        app,
        [
            "compare",
            str(dataset),
            "-p",
            f"sharp={sharp}",
            "-p",
            f"soft={soft}",
            "--cal-ratio",
            "1.0",
        ],
    )

    assert result.exit_code != 0
    assert "cal-ratio" in plain_output(result).lower()


def test_compare_applies_temperature_when_calibrate(tmp_path):
    dataset, sharp, soft = _write_two_models(tmp_path, n=240)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "compare",
            str(dataset),
            "-p",
            f"sharp={sharp}",
            "-p",
            f"soft={soft}",
            "--output",
            str(out),
            "--target-risks",
            "0.10",
            "--temp-ratio",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((out / "compare_report.json").read_text(encoding="utf-8"))
    temps = [e.get("temperature") for e in payload["entries"]]
    assert all(t is not None and t > 0 for t in temps)
