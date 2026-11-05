"""Ordinal under-rate loss on contracts with choice_order."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal.certify import run_certification
from decisionops.conformal.risk_control import ORDINAL_QUANTITY, certify_threshold
from decisionops.contract import DecisionContract
from decisionops.decide import decide_many, load_profile, profile_from_report, save_profile
from decisionops.recertify import run_recertification
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

runner = CliRunner()

LEVELS = ["skip", "shallow", "deep", "human"]

ORDINAL = DecisionContract(
    name="ordinal-routing",
    choices=LEVELS,
    auto_choices=LEVELS,
    choice_order=LEVELS,
    under_rate_tolerance=1,
)


def _pred(cid: str, choice: str, conf: float) -> DecisionPrediction:
    rest = (1.0 - conf) / (len(LEVELS) - 1)
    probs = {c: rest for c in LEVELS}
    probs[choice] = conf
    return DecisionPrediction(id=cid, prediction=choice, probabilities=probs, model="ord-v1")


def test_miss_loss_only_under_rate_beyond_tolerance():
    # miss_loss(prediction, label); ranks skip=0..human=3; k=1 → error iff gap > 1
    assert ORDINAL.miss_loss("deep", "deep") == 0.0
    assert ORDINAL.miss_loss("deep", "shallow") == 0.0  # over-rate free
    assert ORDINAL.miss_loss("deep", "human") == 0.0  # under by 1, allowed
    assert ORDINAL.miss_loss("shallow", "human") == 1.0  # under by 2
    assert ORDINAL.miss_loss("skip", "human") == 1.0
    assert ORDINAL.miss_loss("skip", "deep") == 1.0  # under by 2
    assert ORDINAL.miss_loss("shallow", "deep") == 0.0  # under by 1


def test_tolerance_zero_flags_any_under_rate():
    c = ORDINAL.model_copy(update={"under_rate_tolerance": 0})
    assert c.miss_loss("shallow", "deep") == 1.0
    assert c.miss_loss("deep", "deep") == 0.0
    assert c.miss_loss("deep", "shallow") == 0.0


def test_schema_rejects_bad_order_and_severity_mix():
    with pytest.raises(ValidationError, match="choice_order"):
        DecisionContract(
            name="x", choices=["a", "b"], auto_choices=["a"], choice_order=["a"]
        )
    with pytest.raises(ValidationError, match="cannot be combined with severity"):
        DecisionContract(
            name="x",
            choices=["a", "b"],
            auto_choices=["a"],
            choice_order=["a", "b"],
            severity={"b": 2.0},
        )


def test_fingerprint_covers_order_and_tolerance():
    a = ORDINAL.fingerprint()
    b = ORDINAL.model_copy(update={"under_rate_tolerance": 0}).fingerprint()
    c = ORDINAL.model_copy(update={"choice_order": list(reversed(LEVELS))}).fingerprint()
    assert a != b and a != c


def test_certify_uses_ordinal_quantity_not_equality():
    """
    Every case is under-rated by exactly one level, same confidence.
    Equality loss is 1 on every auto; ordinal k=1 loss is 0. Threshold cannot
    filter by confidence, so equality is infeasible and ordinal covers all.
    """
    cases, preds = [], []
    for i in range(200):
        true_i = 1 + (i % 3)  # shallow..human
        pred_i = true_i - 1
        cases.append(DecisionCase(id=str(i), choices=LEVELS, label=LEVELS[true_i]))
        preds.append(_pred(str(i), LEVELS[pred_i], 0.9))

    pairs = list(zip(cases, preds, strict=True))
    eq = certify_threshold(pairs, 0.05, auto_choices=ORDINAL.auto_set)
    ord_c = certify_threshold(
        pairs,
        0.05,
        auto_choices=ORDINAL.auto_set,
        loss_of=lambda case, pred: ORDINAL.miss_loss(pred.prediction, case.label),
        controlled_quantity=ORDINAL_QUANTITY,
    )
    assert eq.feasible is False
    assert ord_c.feasible
    assert ord_c.controlled_quantity == ORDINAL_QUANTITY
    assert ord_c.empirical_joint_error == 0.0
    assert ord_c.coverage_calibration == 1.0


def test_run_certification_and_decide_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    cases, preds = [], []
    for i in range(600):
        true_i = int(rng.integers(0, 4))
        # Mostly exact or mild under; occasional bad under-rate
        if rng.random() < 0.85:
            pred_i = true_i if rng.random() < 0.7 else max(0, true_i - 1)
        else:
            pred_i = max(0, true_i - 2)
        conf = float(rng.uniform(0.55, 0.99))
        cases.append(DecisionCase(id=str(i), choices=LEVELS, label=LEVELS[true_i]))
        preds.append(_pred(str(i), LEVELS[pred_i], conf))

    report = run_certification(
        cases, preds, alphas=[0.10], contract=ORDINAL, calibrate=False, seed=2
    )
    level = report.levels[0]
    assert level.certificate.feasible
    assert level.certificate.controlled_quantity == ORDINAL_QUANTITY
    profile = profile_from_report(report, 0.10, audit_rate=0.2, audit_seed=3)
    profile = profile.model_copy(update={"model": "ord-v1"})
    assert profile.controlled_quantity == ORDINAL_QUANTITY

    decisions = decide_many(preds, profile)
    assert sum(d.action == "auto" for d in decisions) > 0

    # Recertify on audit labels with same ordinal contract
    outcomes = [OutcomeFeedback(id=c.id, label=c.label) for c in cases]
    rec = run_recertification(
        profile,
        decisions=decisions,
        outcomes=outcomes,
        predictions=preds,
        contract=ORDINAL,
    )
    assert rec.verdict in {"keep", "replace"}
    assert rec.certificate.feasible
    assert rec.certificate.controlled_quantity == ORDINAL_QUANTITY
    assert rec.profile is not None
    assert rec.profile.controlled_quantity == ORDINAL_QUANTITY

    path = save_profile(profile, tmp_path / "ord.json")
    loaded = load_profile(path)
    assert loaded.controlled_quantity == ORDINAL_QUANTITY


def test_cli_certify_ordinal_contract(tmp_path):
    rng = np.random.default_rng(4)
    ds, pr = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    d_lines, p_lines = [], []
    for i in range(500):
        true_i = int(rng.integers(0, 4))
        pred_i = true_i if rng.random() < 0.8 else max(0, true_i - 1)
        conf = float(rng.uniform(0.6, 0.99))
        d_lines.append(
            json.dumps({"id": str(i), "choices": LEVELS, "label": LEVELS[true_i]})
        )
        p_lines.append(
            json.dumps(
                {
                    "id": str(i),
                    "prediction": LEVELS[pred_i],
                    "probabilities": {
                        c: (conf if c == LEVELS[pred_i] else (1 - conf) / 3) for c in LEVELS
                    },
                    "model": "ord-v1",
                }
            )
        )
    ds.write_text("\n".join(d_lines) + "\n")
    pr.write_text("\n".join(p_lines) + "\n")
    contract = tmp_path / "c.json"
    contract.write_text(ORDINAL.model_dump_json())
    out = tmp_path / "out"
    profile = tmp_path / "profile.json"
    result = runner.invoke(
        app,
        [
            "certify",
            str(ds),
            "-p",
            str(pr),
            "--contract",
            str(contract),
            "--alphas",
            "0.1",
            "--no-calibrate",
            "--profile",
            str(profile),
            "--profile-alpha",
            "0.1",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ordinal" in result.output.lower() or profile.exists()
    loaded = load_profile(profile)
    assert loaded.controlled_quantity == ORDINAL_QUANTITY


def test_selective_rejects_ordinal_contract():
    cases = [DecisionCase(id="0", choices=LEVELS, label="deep")]
    preds = [_pred("0", "deep", 0.9)]
    with pytest.raises(ValueError, match="ordinal"):
        run_certification(
            cases, preds, alphas=[0.1], contract=ORDINAL, risk="selective", calibrate=False
        )
