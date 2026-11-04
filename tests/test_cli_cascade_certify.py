"""CLI `certify-cascade` and cascade `decide`, plus test-split replay."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal.cascade_certify import run_cascade_certification
from decisionops.conformal.cascade_risk import stage_arrays_from_pairs
from decisionops.contract import DecisionContract
from decisionops.decide import cascade_profile_from_report, decide_cascade_many, load_profile
from decisionops.io import align_cases_and_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction
from decisionops.split import split_pairs

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
)


def _write_two_stage(tmp_path, n: int = 500):
    """
    Stage 1 is confident and usually right; stage 2 is a bit flatter and
    catches some of what stage 1 misses. Large enough for LTT at 5%, 90%.
    """
    dataset = tmp_path / "dataset.jsonl"
    cheap = tmp_path / "cheap.jsonl"
    mid = tmp_path / "mid.jsonl"
    ds, c_lines, m_lines = [], [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        cid = str(i)
        c1 = float(rng.uniform(0.55, 0.999))
        c2 = float(rng.uniform(0.70, 0.999))
        ok1 = rng.random() < c1
        ok2 = rng.random() < c2
        label = "skip" if (ok1 or ok2) else "deep"
        ds.append(json.dumps({"id": cid, "choices": ["skip", "deep"], "label": label}))
        c_lines.append(
            json.dumps(
                {
                    "id": cid,
                    "prediction": "skip",
                    "probabilities": {"skip": c1, "deep": 1.0 - c1},
                    "model": "cheap",
                }
            )
        )
        m_lines.append(
            json.dumps(
                {
                    "id": cid,
                    "prediction": "skip",
                    "probabilities": {"skip": c2, "deep": 1.0 - c2},
                    "model": "mid",
                }
            )
        )
    dataset.write_text("\n".join(ds) + "\n", encoding="utf-8")
    cheap.write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    mid.write_text("\n".join(m_lines) + "\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(ROUTING.model_dump_json(), encoding="utf-8")
    return dataset, cheap, mid, contract


def test_certify_cascade_cli_writes_report_and_profile(tmp_path):
    dataset, cheap, mid, contract = _write_two_stage(tmp_path)
    out = tmp_path / "out"
    profile = tmp_path / "cascade.json"
    result = runner.invoke(
        app,
        [
            "certify-cascade",
            str(dataset),
            "-p",
            f"cheap={cheap}",
            "-p",
            f"mid={mid}",
            "--contract",
            str(contract),
            "--alpha",
            "0.05",
            "--no-calibrate",
            "--output",
            str(out),
            "--profile",
            str(profile),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cascade Certification" in result.output
    assert "joint" in result.output.lower()
    payload = json.loads((out / "cascade_certification_report.json").read_text())
    assert payload["metadata"]["protocol"] == "three_way_split_temp_cascade_ltt_test"
    assert payload["certificate"]["guarantee"] == "high_probability"
    loaded = load_profile(profile)
    assert loaded.kind == "cascade"
    assert len(loaded.stages) == 2


def test_certify_cascade_rejects_one_prediction_set(tmp_path):
    dataset, cheap, _, contract = _write_two_stage(tmp_path, n=80)
    result = runner.invoke(
        app,
        [
            "certify-cascade",
            str(dataset),
            "-p",
            f"cheap={cheap}",
            "--contract",
            str(contract),
        ],
    )
    assert result.exit_code != 0
    assert "exactly two" in result.output.lower()


def test_decide_cli_cascade_round_trip(tmp_path):
    dataset, cheap, mid, contract = _write_two_stage(tmp_path)
    profile = tmp_path / "cascade.json"
    certify = runner.invoke(
        app,
        [
            "certify-cascade",
            str(dataset),
            "-p",
            f"cheap={cheap}",
            "-p",
            f"mid={mid}",
            "--contract",
            str(contract),
            "--no-calibrate",
            "--profile",
            str(profile),
        ],
    )
    assert certify.exit_code == 0, certify.output
    out = tmp_path / "decisions.jsonl"
    decide = runner.invoke(
        app,
        [
            "decide",
            str(profile),
            "-p",
            f"cheap={cheap}",
            "-p",
            f"mid={mid}",
            "--contract",
            str(contract),
            "-o",
            str(out),
        ],
    )
    assert decide.exit_code == 0, decide.output
    assert "lambda1=" in decide.output
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert {r["action"] for r in rows} <= {"auto", "escalate"}
    assert any(r.get("stage") in (1, 2, None) for r in rows)


def test_decide_replays_test_split_counts(tmp_path):
    """decide_cascade_many on the held-out split matches the certificate's test point."""
    dataset, cheap, mid, _ = _write_two_stage(tmp_path, n=600)
    cases = [
        DecisionCase.model_validate(json.loads(line))
        for line in dataset.read_text().splitlines()
        if line
    ]
    p1 = [
        DecisionPrediction.model_validate(json.loads(line))
        for line in cheap.read_text().splitlines()
        if line
    ]
    p2 = [
        DecisionPrediction.model_validate(json.loads(line))
        for line in mid.read_text().splitlines()
        if line
    ]
    report = run_cascade_certification(
        cases, p1, p2, ROUTING, calibrate=False, alpha=0.05, seed=42
    )
    assert report.certificate.feasible
    assert report.test is not None
    profile = cascade_profile_from_report(report)

    pairs1 = align_cases_and_predictions(cases, p1)
    split = split_pairs(pairs1, cal_ratio=0.3, seed=42)
    test_ids = {c.id for c, _ in split.test}
    t1 = [p for p in p1 if p.id in test_ids]
    t2 = [p for p in p2 if p.id in test_ids]
    decisions = decide_cascade_many(t1, profile, t2)
    n_auto = sum(d.action == "auto" for d in decisions)
    # Replay without labels: count autos. Errors need labels.
    assert n_auto == report.test.n_auto_stage1 + report.test.n_auto_stage2

    by_id = {c.id: c for c, _ in split.test}
    pred1 = {p.id: p for p in t1}
    pred2 = {p.id: p for p in t2}
    n_err = 0
    for d in decisions:
        if d.action != "auto":
            continue
        case = by_id[d.id]
        chosen = pred1[d.id] if d.stage == 1 else pred2[d.id]
        if chosen.prediction != case.label:
            n_err += 1
    assert n_err == report.test.n_errors


def test_stage_arrays_hide_escalate_choices():
    case = DecisionCase(id="1", choices=["skip", "deep"], label="deep")
    pred = DecisionPrediction(
        id="1", prediction="deep", probabilities={"deep": 0.99, "skip": 0.01}
    )
    arr = stage_arrays_from_pairs([(case, pred)], auto_choices=frozenset({"skip"}))
    assert arr.confidence[0] == float("-inf")


def test_run_cascade_fits_temperature_and_prices_stages(tmp_path):
    dataset, cheap, mid, _ = _write_two_stage(tmp_path, n=600)
    cases = [
        DecisionCase.model_validate(json.loads(line))
        for line in dataset.read_text().splitlines()
        if line
    ]
    p1 = [
        DecisionPrediction.model_validate(json.loads(line))
        for line in cheap.read_text().splitlines()
        if line
    ]
    p2 = [
        DecisionPrediction.model_validate(json.loads(line))
        for line in mid.read_text().splitlines()
        if line
    ]
    report = run_cascade_certification(
        cases,
        p1,
        p2,
        ROUTING,
        alpha=0.05,
        calibrate=True,
        stage_costs=(0.001, 0.01),
        seed=1,
    )
    assert report.stages[0].temperature is not None
    assert report.metadata["grid_source"] == "temperature_slice"
    if report.certificate.feasible:
        assert report.certificate.objective == "cost"
        assert report.test is not None
        assert report.test.mean_cost is not None


def test_run_cascade_rejects_severity():
    weighted = DecisionContract(
        name="w",
        choices=["skip", "deep"],
        auto_choices=["skip"],
        severity={"deep": 3.0},
    )
    cases = [DecisionCase(id="1", choices=["skip", "deep"], label="skip")]
    pred = DecisionPrediction(
        id="1", prediction="skip", probabilities={"skip": 0.9, "deep": 0.1}
    )
    with pytest.raises(ValueError, match="0-1 loss"):
        run_cascade_certification(cases * 10, [pred] * 10, [pred] * 10, weighted)
