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
    audit_draw,
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


def test_audit_off_by_default_changes_nothing():
    cases, preds = _pool(n=400)
    plain = decide_many(preds, _profile())
    assert not any(d.audit for d in plain)
    assert all(d.reason != "audit" for d in plain)


def test_audit_draw_rate_is_reproducible_and_seeded():
    ids = [f"case-{i}" for i in range(20_000)]
    a = [audit_draw(7, i) < 0.05 for i in ids]
    assert sum(a) / len(ids) == pytest.approx(0.05, abs=0.006)
    assert a == [audit_draw(7, i) < 0.05 for i in ids]  # stateless, reproducible
    b = [audit_draw(8, i) < 0.05 for i in ids]
    assert a != b  # the seed moves the sample


def test_audit_forces_escalation_and_marks_escalations_too():
    p = _profile(audit_rate=0.5, audit_seed=3)
    drawn = [i for i in map(str, range(200)) if audit_draw(3, i) < 0.5]
    not_drawn = [i for i in map(str, range(200)) if audit_draw(3, i) >= 0.5]

    d = decide(_pred("skip", 0.99, pid=drawn[0]), p)
    assert (d.action, d.reason, d.audit) == ("escalate", "audit", True)
    # Already escalating: keeps its own reason, still in the audit sample.
    d = decide(_pred("skip", 0.5, pid=drawn[0]), p)
    assert (d.action, d.reason, d.audit) == ("escalate", "below_threshold", True)
    d = decide(_pred("deep", 0.99, pid=drawn[0]), p)
    assert (d.reason, d.audit) == ("escalate_choice", True)

    d = decide(_pred("skip", 0.99, pid=not_drawn[0]), p)
    assert (d.action, d.reason, d.audit) == ("auto", "certified", False)


def test_audit_sample_is_independent_of_confidence():
    """The audit set must look like all traffic, unlike organic escalations."""
    from scipy.stats import ks_2samp

    _, preds = _pool(n=6000, seed=11)
    p = _profile(audit_rate=0.1, audit_seed=1)
    decisions = decide_many(preds, p)
    conf_all = np.array([d.confidence for d in decisions])
    conf_audit = np.array([d.confidence for d in decisions if d.audit])
    conf_organic = np.array(
        [d.confidence for d in decisions if d.action == "escalate" and not d.audit]
    )
    assert 450 < conf_audit.size < 750
    assert ks_2samp(conf_audit, conf_all).pvalue > 0.01
    # Contrast: organic escalations are visibly selected by confidence.
    assert ks_2samp(conf_organic, conf_all).pvalue < 1e-6


def test_audit_only_removes_automated_errors():
    cases, preds = _pool(n=2000, seed=5)
    labels = {c.id: c.label for c in cases}
    base = decide_many(preds, _profile())
    audited = decide_many(preds, _profile(audit_rate=0.2, audit_seed=9))

    def auto_errors(ds):
        return {d.id for d in ds if d.action == "auto" and d.choice != labels[d.id]}

    assert auto_errors(audited) <= auto_errors(base)
    n_base = sum(d.action == "auto" for d in base)
    n_aud = sum(d.action == "auto" for d in audited)
    assert n_aud == pytest.approx(0.8 * n_base, rel=0.1)


def test_audit_rate_round_trips_and_is_validated(tmp_path):
    p = _profile(audit_rate=0.05, audit_seed=42)
    path = save_profile(p, tmp_path / "p.json")
    loaded = load_profile(path)
    assert (loaded.audit_rate, loaded.audit_seed) == (0.05, 42)
    assert loaded.profile_hash == p.profile_hash
    assert _profile().profile_hash != p.profile_hash  # audit settings are part of the id
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            _profile(audit_rate=bad)


def test_cli_certify_audit_rate_then_decide(tmp_path):
    cases, preds = _pool()
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    contract = tmp_path / "contract.json"
    _write_jsonl(dataset, [c.model_dump(mode="json") for c in cases])
    _write_jsonl(predictions, [p.model_dump(mode="json") for p in preds])
    contract.write_text(ROUTING.model_dump_json(), encoding="utf-8")
    profile = tmp_path / "profile.json"
    base = ["certify", str(dataset), "-p", str(predictions), "--contract", str(contract),
            "--alphas", "0.10"]

    result = runner.invoke(app, [*base, "--audit-rate", "0.05"])
    assert result.exit_code != 0  # needs --profile
    result = runner.invoke(app, [*base, "--profile", str(profile), "--audit-rate", "1.0"])
    assert result.exit_code != 0

    result = runner.invoke(
        app, [*base, "--profile", str(profile), "--audit-rate", "0.1", "--audit-seed", "4"]
    )
    assert result.exit_code == 0, result.output
    assert (load_profile(profile).audit_rate, load_profile(profile).audit_seed) == (0.1, 4)

    out = tmp_path / "decisions.jsonl"
    result = runner.invoke(app, ["decide", str(profile), "-p", str(predictions), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "audit sample:" in result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    n_audit = sum(r["audit"] for r in rows)
    assert 0 < n_audit < len(rows)
    assert all(r["action"] == "escalate" for r in rows if r["audit"])


def test_cli_decide_warns_on_duplicate_ids(tmp_path):
    profile = save_profile(_profile(audit_rate=0.1), tmp_path / "p.json")
    preds = [_pred("skip", 0.95, pid=i) for i in ("a", "b", "a", "c", "c", "c")]
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [p.model_dump(mode="json") for p in preds])
    result = runner.invoke(app, ["decide", str(profile), "-p", str(predictions)])
    assert result.exit_code == 0, result.output
    assert "2 id(s) appear more than once" in result.output
    assert "'a', 'c'" in result.output

    unique = tmp_path / "unique.jsonl"
    _write_jsonl(unique, [_pred("skip", 0.95, pid=i).model_dump(mode="json") for i in "abc"])
    result = runner.invoke(app, ["decide", str(profile), "-p", str(unique)])
    assert "more than once" not in result.output
