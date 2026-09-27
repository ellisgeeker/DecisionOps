"""Certified profile + runtime decide()."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from decisionops.calibration.temperature import apply_temperature
from decisionops.cli.main import app
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract
from decisionops.decide import (
    CertificationProfile,
    ProfileMismatchError,
    decide,
    decide_many,
    load_profile,
    profile_from_report,
    save_profile,
)
from decisionops.io import align_cases_and_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction
from decisionops.split import split_pairs

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
    descriptions={"skip": "safe", "deep": "risky"},
)


def _pool(n: int = 600, seed: int = 0):
    """Noisy binary routing pool: confidence correlates with correctness, imperfectly."""
    rng = np.random.default_rng(seed)
    cases, preds = [], []
    for i in range(n):
        pred = "skip" if rng.random() < 0.7 else "deep"
        conf = float(rng.uniform(0.5, 0.999))
        correct = rng.random() < conf**2
        label = pred if correct else ("deep" if pred == "skip" else "skip")
        other = "deep" if pred == "skip" else "skip"
        cases.append(DecisionCase(id=str(i), choices=["skip", "deep"], label=label))
        preds.append(
            DecisionPrediction(
                id=str(i),
                prediction=pred,
                probabilities={pred: conf, other: 1.0 - conf},
                provider="static",
                model="fixture-v1",
            )
        )
    return cases, preds


def _profile(**overrides) -> CertificationProfile:
    base = {
        "alpha": 0.05,
        "threshold": 0.9,
        "model": "fixture-v1",
        "choices": ["skip", "deep"],
        "auto_choices": ["skip"],
        "contract_hash": ROUTING.fingerprint(),
        "n_calibration": 300,
    }
    return CertificationProfile(**{**base, **overrides})


def _pred(choice: str, conf: float, model: str | None = "fixture-v1", pid: str = "x"):
    other = "deep" if choice == "skip" else "skip"
    return DecisionPrediction(
        id=pid, prediction=choice, probabilities={choice: conf, other: 1.0 - conf}, model=model
    )


def test_decide_reasons():
    p = _profile()
    assert decide(_pred("skip", 0.95), p).reason == "certified"
    assert decide(_pred("skip", 0.95), p).action == "auto"
    assert decide(_pred("skip", 0.90), p).action == "auto"  # >= threshold, like CRC
    assert decide(_pred("skip", 0.89), p).reason == "below_threshold"
    # High confidence in an escalation choice is still an escalation.
    d = decide(_pred("deep", 0.99), p)
    assert (d.action, d.reason) == ("escalate", "escalate_choice")


def test_decide_escalates_choice_outside_contract():
    p = _profile()
    pred = DecisionPrediction(id="x", prediction="nuke", probabilities={"nuke": 0.99, "skip": 0.01})
    d = decide(pred, p)
    assert (d.action, d.reason) == ("escalate", "outside_contract")


def test_decide_without_contract_treats_every_choice_as_auto():
    p = _profile(choices=None, auto_choices=None, contract_hash=None)
    assert decide(_pred("deep", 0.95), p).action == "auto"


def test_decide_applies_temperature_before_threshold():
    # Raw 0.95 softened by T=3 lands well below 0.9.
    p = _profile(temperature=3.0)
    d = decide(_pred("skip", 0.95), p)
    expected = apply_temperature(_pred("skip", 0.95), 3.0).confidence
    assert d.confidence == pytest.approx(expected)
    assert d.action == "escalate"


def test_decide_rejects_other_model():
    with pytest.raises(ProfileMismatchError, match="profile certifies"):
        decide(_pred("skip", 0.99, model="other-v2"), _profile())
    # Unlabelled predictions are accepted: model tags are optional in JSONL.
    assert decide(_pred("skip", 0.99, model=None), _profile()).action == "auto"


def test_check_contract_catches_changed_model_input():
    p = _profile()
    p.check_contract(ROUTING)
    p.check_contract(None)
    # Costs do not change the certificate.
    p.check_contract(ROUTING.model_copy(update={"costs": {"skip": 0.0, "deep": 9.0}}))
    reworded = ROUTING.model_copy(update={"descriptions": {"skip": "safe!", "deep": "risky"}})
    with pytest.raises(ProfileMismatchError, match="recertify"):
        p.check_contract(reworded)
    widened = ROUTING.model_copy(update={"auto_choices": ["skip", "deep"]})
    with pytest.raises(ProfileMismatchError):
        p.check_contract(widened)
    with pytest.raises(ProfileMismatchError, match="without a contract"):
        _profile(contract_hash=None).check_contract(ROUTING)


def test_decide_reproduces_certified_test_numbers():
    """
    decide() on the held-out split must match the certificate's own test stats.

    If this drifts, the runtime rule is no longer the rule CRC certified.
    """
    cases, preds = _pool()
    report = run_certification(
        cases, preds, alphas=[0.05, 0.10], contract=ROUTING, seed=7, calibrate=True
    )
    for alpha in (0.05, 0.10):
        profile = profile_from_report(report, alpha)
        assert profile.temperature is not None
        assert profile.contract_hash == ROUTING.fingerprint()

        # Same split as run_certification; test rows carry raw probabilities.
        test_pairs = split_pairs(
            align_cases_and_predictions(cases, preds), cal_ratio=0.3, seed=7
        ).test
        decisions = decide_many([p for _, p in test_pairs], profile)
        label = {c.id: c.label for c, _ in test_pairs}

        n_auto = sum(d.action == "auto" for d in decisions)
        n_miss = sum(d.action == "auto" and d.choice != label[d.id] for d in decisions)
        assert profile.test is not None
        assert n_auto == profile.test["n_accepted"]
        assert n_miss / len(decisions) == pytest.approx(profile.test["joint_error_rate"])
        assert all(d.choice == "skip" for d in decisions if d.action == "auto")


def test_profile_from_report_rejects_missing_or_infeasible_alpha():
    cases, preds = _pool(n=120)
    report = run_certification(cases, preds, alphas=[0.01, 0.10], contract=ROUTING)
    with pytest.raises(ValueError, match="not in report"):
        profile_from_report(report, 0.05)
    # n_crc_cal ~ 29 < 99, so 1% cannot be certified.
    with pytest.raises(ValueError, match="no certified threshold"):
        profile_from_report(report, 0.01)


def test_profile_round_trip_and_schema_guard(tmp_path):
    p = _profile(temperature=1.3)
    path = save_profile(p, tmp_path / "p.json")
    loaded = load_profile(path)
    assert loaded == p
    assert loaded.profile_hash == p.profile_hash

    raw = json.loads(path.read_text())
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema_version"):
        load_profile(path)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_cli_certify_profile_then_decide(tmp_path):
    cases, preds = _pool()
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    contract = tmp_path / "contract.json"
    _write_jsonl(dataset, [c.model_dump(mode="json") for c in cases])
    _write_jsonl(predictions, [p.model_dump(mode="json") for p in preds])
    contract.write_text(ROUTING.model_dump_json(), encoding="utf-8")
    profile = tmp_path / "profile.json"

    result = runner.invoke(
        app,
        [
            "certify", str(dataset), "-p", str(predictions), "--contract", str(contract),
            "--alphas", "0.05,0.10", "--profile", str(profile), "--profile-alpha", "0.10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert load_profile(profile).alpha == 0.10

    out = tmp_path / "decisions.jsonl"
    result = runner.invoke(
        app,
        ["decide", str(profile), "-p", str(predictions), "--contract", str(contract),
         "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "auto=" in result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(preds)
    assert {r["action"] for r in rows} <= {"auto", "escalate"}

    # A reworded contract is refused at decide time.
    changed = ROUTING.model_copy(update={"instructions": "Different question?"})
    contract.write_text(changed.model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app, ["decide", str(profile), "-p", str(predictions), "--contract", str(contract)]
    )
    assert result.exit_code != 0
    assert "recertify" in result.output


def test_cli_certify_profile_argument_checks(tmp_path):
    cases, preds = _pool(n=200)
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset, [c.model_dump(mode="json") for c in cases])
    _write_jsonl(predictions, [p.model_dump(mode="json") for p in preds])
    base = ["certify", str(dataset), "-p", str(predictions)]

    result = runner.invoke(app, [*base, "--profile", str(tmp_path / "p.json")])
    assert result.exit_code != 0  # several alphas, no --profile-alpha
    result = runner.invoke(app, [*base, "--profile-alpha", "0.05"])
    assert result.exit_code != 0  # --profile-alpha alone
    result = runner.invoke(
        app, [*base, "--profile", str(tmp_path / "p.json"), "--profile-alpha", "0.2"]
    )
    assert result.exit_code != 0  # not in --alphas

    # Single infeasible alpha: report prints, no profile, non-zero exit.
    result = runner.invoke(
        app, [*base, "--alphas", "0.001", "--profile", str(tmp_path / "p.json")]
    )
    assert result.exit_code == 1
    assert not (tmp_path / "p.json").exists()
