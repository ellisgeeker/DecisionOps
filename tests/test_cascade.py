"""Offline cascade / cost–miss curve."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from decisionops.cascade import cost_curve, escalate_cost, run_cascade
from decisionops.cli.main import app
from decisionops.contract import DecisionContract
from decisionops.schemas import DecisionCase, DecisionPrediction

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
)


def _pair(cid: str, pred: str, label: str, conf: float):
    other = "deep" if pred == "skip" else "skip"
    return (
        DecisionCase(id=cid, choices=["skip", "deep"], label=label),
        DecisionPrediction(
            id=cid, prediction=pred, probabilities={pred: conf, other: 1.0 - conf}
        ),
    )


def _pool():
    # 8 skip-correct (high conf), 1 skip-wrong (mid conf), 1 deep-label (model says skip, low conf)
    pairs = []
    for i in range(8):
        pairs.append(_pair(f"ok{i}", "skip", "skip", 0.95))
    pairs.append(_pair("miss", "skip", "deep", 0.70))
    pairs.append(_pair("need", "skip", "deep", 0.40))
    return pairs


def test_escalate_cost_prefers_true_escalate_label():
    case = DecisionCase(id="1", choices=["skip", "deep"], label="deep")
    assert escalate_cost(ROUTING, case) == 1.0
    safe = DecisionCase(id="2", choices=["skip", "deep"], label="skip")
    assert escalate_cost(ROUTING, safe) == 1.0  # cheapest escalate


def test_cost_curve_monotone_in_threshold():
    pairs = _pool()
    baseline, always_cost, always_miss, curve = cost_curve(
        pairs, ROUTING, thresholds=[0.0, 0.5, 0.8, 0.90]
    )
    assert baseline == pytest.approx(1.0)
    # Raising the threshold never increases automation.
    autos = [p.automation_rate for p in curve]
    assert autos == sorted(autos, reverse=True)
    # Always-auto (t=0) should automate every "skip" prediction.
    assert curve[0].n_auto == 10
    assert always_miss == pytest.approx(curve[0].miss_rate)
    assert always_cost == pytest.approx(curve[0].mean_cost)
    # Strict threshold automates only the high-confidence skips → lower miss, less saving.
    strict = curve[-1]
    assert strict.n_auto == 8
    assert strict.miss_rate == pytest.approx(0.0)
    assert strict.cost_saved < curve[0].cost_saved


def test_run_cascade_requires_full_costs_and_splits():
    with pytest.raises(ValueError, match="missing costs"):
        run_cascade(
            [DecisionCase(id="1", choices=["skip", "deep"], label="skip")],
            [
                DecisionPrediction(
                    id="1", prediction="skip", probabilities={"skip": 0.9, "deep": 0.1}
                )
            ],
            DecisionContract(name="x", choices=["skip", "deep"], auto_choices=["skip"]),
        )

    pairs = []
    for i in range(5):
        for case, pred in _pool():
            cid = f"{case.id}-{i}"
            pairs.append(
                (
                    case.model_copy(update={"id": cid}),
                    pred.model_copy(update={"id": cid}),
                )
            )
    cases = [c for c, _ in pairs]
    preds = [p for _, p in pairs]
    report = run_cascade(cases, preds, ROUTING, cal_ratio=0.3, seed=0)
    assert report.split.n_test >= 2
    assert report.curve
    assert report.metadata["escalation_pricing"] == "oracle"
    assert report.contract["name"] == "routing"
    # cost_saved = baseline - mean_cost
    for point in report.curve:
        assert point.cost_saved == pytest.approx(
            report.baseline_mean_cost - point.mean_cost
        )


def test_cascade_cli_writes_reports(tmp_path):
    pairs = []
    for i in range(8):
        for case, pred in _pool():
            cid = f"{case.id}-{i}"
            pairs.append(
                (
                    case.model_copy(update={"id": cid}),
                    pred.model_copy(update={"id": cid}),
                )
            )
    ds = tmp_path / "dataset.jsonl"
    pr = tmp_path / "predictions.jsonl"
    contract_path = tmp_path / "contract.json"
    ds_lines, pr_lines = [], []
    for case, pred in pairs:
        ds_lines.append(
            json.dumps({"id": case.id, "choices": case.choices, "label": case.label})
        )
        pr_lines.append(
            json.dumps(
                {
                    "id": pred.id,
                    "prediction": pred.prediction,
                    "probabilities": pred.probabilities,
                    "provider": "static",
                    "model": "fixture",
                }
            )
        )
    ds.write_text("\n".join(ds_lines) + "\n", encoding="utf-8")
    pr.write_text("\n".join(pr_lines) + "\n", encoding="utf-8")
    contract_path.write_text(
        json.dumps(
            {
                "name": "routing",
                "choices": ["skip", "deep"],
                "auto_choices": ["skip"],
                "costs": {"skip": 0.0, "deep": 1.0},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "cascade",
            str(ds),
            "--predictions",
            str(pr),
            "--contract",
            str(contract_path),
            "--output",
            str(out),
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cost vs miss rate" in result.output.lower() or "Cost–miss" in result.output
    assert (out / "cascade_report.json").exists()
    assert (out / "cascade_report.html").exists()
    payload = json.loads((out / "cascade_report.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["protocol"] == "cal_test_cost_curve"
    assert len(payload["curve"]) >= 2
    html = (out / "cascade_report.html").read_text(encoding="utf-8")
    assert "Cost saved vs miss rate" in html


def test_cascade_cli_rejects_contract_without_costs(tmp_path):
    ds = tmp_path / "dataset.jsonl"
    pr = tmp_path / "predictions.jsonl"
    contract_path = tmp_path / "contract.json"
    ds.write_text(
        json.dumps({"id": "1", "choices": ["a", "b"], "label": "a"}) + "\n",
        encoding="utf-8",
    )
    pr.write_text(
        json.dumps(
            {"id": "1", "prediction": "a", "probabilities": {"a": 0.9, "b": 0.1}}
        )
        + "\n",
        encoding="utf-8",
    )
    contract_path.write_text(
        json.dumps({"name": "x", "choices": ["a", "b"], "auto_choices": ["a"]}),
        encoding="utf-8",
    )
    # Need enough rows for split — pad
    for path, maker in (
        (
            ds,
            lambda i: json.dumps({"id": str(i), "choices": ["a", "b"], "label": "a"}),
        ),
        (
            pr,
            lambda i: json.dumps(
                {
                    "id": str(i),
                    "prediction": "a",
                    "probabilities": {"a": 0.9, "b": 0.1},
                }
            ),
        ),
    ):
        lines = [maker(i) for i in range(20)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "cascade",
            str(ds),
            "--predictions",
            str(pr),
            "--contract",
            str(contract_path),
        ],
    )
    assert result.exit_code != 0
    assert "cost" in result.output.lower()
