"""
Certified selective risk (Learn-Then-Test): error rate AMONG auto-accepted cases.

The pool is calibrated, c ~ U(LO, HI) and P(correct | c) = c, so the true
selective risk of accepting c >= lam is the mean of (1 - c) over U(lam, HI):

    selective_risk(lam) = 1 - (lam + HI) / 2

The checks below measure the guarantee against that closed form, not against a
noisy test split.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal import certify_threshold, run_certification
from decisionops.conformal.risk_control import (
    SELECTIVE_QUANTITY,
    certify_selective_threshold,
    min_selective_accepted,
)
from decisionops.contract import DecisionContract
from decisionops.decide import decide_many, profile_from_report
from decisionops.recertify import run_recertification
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

LO, HI = 0.55, 0.999
runner = CliRunner()


def selective_risk(lam: float) -> float:
    lam = min(max(lam, LO), HI)
    return 1.0 - (lam + HI) / 2.0


def _pool(rng: np.random.Generator, n: int, prefix: str = ""):
    pairs = []
    for i in range(n):
        conf = float(rng.uniform(LO, HI))
        ok = bool(rng.random() < conf)
        cid = f"{prefix}{i}"
        pairs.append(
            (
                DecisionCase(id=cid, choices=["a", "b"], label="a" if ok else "b"),
                DecisionPrediction(
                    id=cid, prediction="a", probabilities={"a": conf, "b": 1.0 - conf}
                ),
            )
        )
    return pairs


def test_min_selective_accepted_is_the_zero_error_floor():
    assert min_selective_accepted(0.05, 0.9) == 45
    assert min_selective_accepted(0.05, 0.95) == 59
    assert min_selective_accepted(0.01, 0.95) == 299


def test_violation_rate_is_at_most_delta():
    """
    The guarantee itself: true selective risk > alpha in at most delta of draws.

    Teeth: this design (alpha=5%, 90%, n=600) is where dropping the fixed
    sequence — testing every lambda and taking the loosest that passes — shows.
    Measured over 500 reps: the real rule violates 1.4%, that mutant 14.9%,
    against delta=10%. Earlier designs (delta=0.2, n=1000) let the mutant pass.
    """
    alpha, level, n, reps = 0.05, 0.9, 600, 500
    viol = feasible = 0
    for rep in range(reps):
        res = certify_selective_threshold(
            _pool(np.random.default_rng(30_000 + rep), n), alpha, confidence_level=level
        )
        if res.feasible:
            feasible += 1
            viol += selective_risk(res.threshold) > alpha
    assert feasible >= 150
    assert viol / feasible <= 1.0 - level, (viol, feasible)


def test_joint_certificate_does_not_answer_the_selective_question():
    """
    The reason this mode exists. Joint CRC at alpha bounds errors over ALL
    traffic, so among what it automates the error rate is far above alpha.
    """
    alpha = 0.05
    joint_sel, ltt_sel = [], []
    for rep in range(60):
        pairs = _pool(np.random.default_rng(31_000 + rep), 1000)
        joint = certify_threshold(pairs, alpha)
        ltt = certify_selective_threshold(pairs, alpha, confidence_level=0.9)
        if joint.feasible:
            joint_sel.append(selective_risk(joint.threshold))
        if ltt.feasible:
            ltt_sel.append(selective_risk(ltt.threshold))
    assert np.mean(joint_sel) > 2 * alpha
    assert all(r > alpha for r in joint_sel)
    assert ltt_sel and np.mean(ltt_sel) <= alpha


def test_scan_start_needs_enough_accepted_cases():
    """
    One very confident correct case must not end the scan at the strictest
    threshold: below the floor even zero errors cannot pass, so the scan starts
    at the first threshold with enough accepted cases.
    """
    pairs = _pool(np.random.default_rng(32_000), 2000)
    res = certify_selective_threshold(pairs, 0.05, confidence_level=0.9)
    assert res.feasible
    assert res.n_accepted_calibration >= min_selective_accepted(0.05, 0.9)
    assert res.controlled_quantity == SELECTIVE_QUANTITY
    assert res.guarantee == "high_probability"


def test_infeasible_when_too_few_auto_cases():
    pairs = _pool(np.random.default_rng(32_001), 30)
    res = certify_selective_threshold(pairs, 0.05, confidence_level=0.9)
    assert not res.feasible
    assert any("needs at least 45" in n for n in res.notes)


def test_run_certification_selective_defaults_and_guards():
    pairs = _pool(np.random.default_rng(33_000), 3000)
    cases, preds = [c for c, _ in pairs], [p for _, p in pairs]
    report = run_certification(cases, preds, alphas=[0.05], calibrate=False, risk="selective")
    cert = report.levels[0].certificate
    assert report.metadata["risk"] == "selective"
    assert cert.confidence_level == 0.9  # selective is high-probability only
    assert cert.controlled_quantity == SELECTIVE_QUANTITY
    # held is judged on the quantity that was certified
    assert report.levels[0].bound_held_on_test == (
        report.levels[0].test["selective_risk"] <= 0.05
    )

    weighted = DecisionContract(
        name="w", choices=["a", "b"], auto_choices=["a"], severity={"b": 3.0}
    )
    with pytest.raises(ValueError, match="severity"):
        run_certification(cases, preds, contract=weighted, risk="selective")
    with pytest.raises(ValueError, match="risk must be"):
        run_certification(cases, preds, risk="conditional")


def test_recertify_keeps_selective_guarantee():
    contract = DecisionContract(name="r", choices=["a", "b"], auto_choices=["a"])
    rng = np.random.default_rng(34_000)
    pairs = _pool(rng, 3000, "pre")
    report = run_certification(
        [c for c, _ in pairs], [p for _, p in pairs],
        alphas=[0.10], contract=contract, calibrate=False, risk="selective",
    )
    # alpha=10%: at 5% some 900-row cal sets are legitimately infeasible.
    profile = profile_from_report(report, 0.10, audit_rate=0.5, audit_seed=1)
    assert profile.controlled_quantity == SELECTIVE_QUANTITY

    live = _pool(rng, 3000, "live")
    decisions = decide_many([p for _, p in live], profile)
    label = {c.id: c.label for c, _ in live}
    outcomes = [OutcomeFeedback(id=d.id, label=label[d.id]) for d in decisions if d.audit]
    rec = run_recertification(
        profile, decisions=decisions, outcomes=outcomes,
        predictions=[p for _, p in live], contract=contract,
    )
    assert rec.certificate.controlled_quantity == SELECTIVE_QUANTITY
    assert rec.profile is not None
    assert rec.profile.controlled_quantity == SELECTIVE_QUANTITY


def test_cli_certify_risk_selective(tmp_path):
    pairs = _pool(np.random.default_rng(35_000), 1500)
    data, preds = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    data.write_text("\n".join(c.model_dump_json() for c, _ in pairs) + "\n")
    preds.write_text("\n".join(p.model_dump_json() for _, p in pairs) + "\n")
    out = runner.invoke(
        app, ["certify", str(data), "-p", str(preds), "--alphas", "0.05",
              "--no-calibrate", "--risk", "selective"],
    )
    assert out.exit_code == 0, out.output
    assert "selective" in out.output
    bad = runner.invoke(app, ["certify", str(data), "-p", str(preds), "--risk", "nope"])
    assert bad.exit_code != 0
