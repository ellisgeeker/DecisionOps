"""CascadeProfile + decide_cascade, and recertify/drift refuse cascade files."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.contract import DecisionContract
from decisionops.decide import (
    CascadeProfile,
    CascadeStage,
    ProfileMismatchError,
    audit_draw,
    cascade_profile_from,
    decide,
    decide_cascade,
    load_profile,
    save_profile,
)
from decisionops.recertify import run_recertification
from decisionops.schemas import DecisionPrediction

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
)


def _pred(choice: str, conf: float, *, model: str, pid: str = "x") -> DecisionPrediction:
    other = "deep" if choice == "skip" else "skip"
    return DecisionPrediction(
        id=pid,
        prediction=choice,
        probabilities={choice: conf, other: 1.0 - conf},
        model=model,
    )


def _profile(**overrides) -> CascadeProfile:
    base = dict(
        alpha=0.05,
        confidence_level=0.9,
        stages=[
            CascadeStage(model="cheap", threshold=0.9),
            CascadeStage(model="mid", threshold=0.85),
        ],
        choices=["skip", "deep"],
        auto_choices=["skip"],
        contract_hash=ROUTING.fingerprint(),
        n_calibration=400,
    )
    base.update(overrides)
    return CascadeProfile(**base)


def test_stage1_auto_does_not_call_stage2():
    called = {"n": 0}

    def get_stage2():
        called["n"] += 1
        raise AssertionError("stage 2 must not run when stage 1 autos")

    d = decide_cascade(
        _pred("skip", 0.95, model="cheap"),
        _profile(),
        get_stage2=get_stage2,
    )
    assert (d.action, d.reason, d.stage) == ("auto", "certified", 1)
    assert called["n"] == 0


def test_stage2_autos_when_stage1_is_below_threshold():
    d = decide_cascade(
        _pred("skip", 0.5, model="cheap"),
        _profile(),
        stage2=_pred("skip", 0.92, model="mid"),
    )
    assert (d.action, d.reason, d.stage) == ("auto", "certified", 2)
    assert d.choice == "skip"
    assert d.threshold == 0.85


def test_both_stages_escalate():
    d = decide_cascade(
        _pred("skip", 0.5, model="cheap"),
        _profile(),
        stage2=_pred("skip", 0.5, model="mid"),
    )
    assert (d.action, d.reason, d.stage) == ("escalate", "below_threshold", None)


def test_escalate_choice_on_stage1_goes_to_stage2():
    """v1: an M1 escalate-choice still asks M2, it does not skip the mid model."""
    d = decide_cascade(
        _pred("deep", 0.99, model="cheap"),
        _profile(),
        stage2=_pred("skip", 0.95, model="mid"),
    )
    assert (d.action, d.reason, d.stage) == ("auto", "certified", 2)


def test_stage_off_skips_that_model():
    p = _profile(
        stages=[
            CascadeStage(model="cheap", threshold=None),
            CascadeStage(model="mid", threshold=0.8),
        ]
    )
    d = decide_cascade(
        _pred("skip", 0.99, model="other"),  # M1 model is not checked when off
        p,
        stage2=_pred("skip", 0.9, model="mid"),
    )
    assert d.stage == 2

    p1 = _profile(
        stages=[
            CascadeStage(model="cheap", threshold=0.9),
            CascadeStage(model="mid", threshold=None),
        ]
    )
    called = {"n": 0}

    def get_stage2():
        called["n"] += 1
        return _pred("skip", 0.99, model="mid")

    d = decide_cascade(_pred("skip", 0.5, model="cheap"), p1, get_stage2=get_stage2)
    assert d.action == "escalate"
    assert called["n"] == 0


def test_model_mismatch_is_per_stage():
    p = _profile()
    with pytest.raises(ProfileMismatchError, match="stage 1"):
        decide_cascade(
            _pred("skip", 0.95, model="wrong"),
            p,
            stage2=_pred("skip", 0.9, model="mid"),
        )
    with pytest.raises(ProfileMismatchError, match="stage 2"):
        decide_cascade(
            _pred("skip", 0.5, model="cheap"),
            p,
            stage2=_pred("skip", 0.9, model="wrong"),
        )


def test_contract_mismatch_and_decide_refuses_cascade():
    p = _profile()
    other = DecisionContract(name="x", choices=["a", "b"], auto_choices=["a"])
    with pytest.raises(ProfileMismatchError):
        p.check_contract(other)
    p.check_contract(ROUTING)

    from decisionops.decide import CertificationProfile

    single = CertificationProfile(
        alpha=0.05, threshold=0.9, n_calibration=10, model="cheap"
    )
    with pytest.raises(ValueError, match="decide_cascade"):
        decide(_pred("skip", 0.95, model="cheap"), p)  # type: ignore[arg-type]
    # single-model decide still works
    assert decide(_pred("skip", 0.95, model="cheap"), single).reason == "certified"


def test_audit_fetches_stage2_even_when_stage1_would_auto():
    p = _profile(audit_rate=0.5, audit_seed=1)
    pid = next(f"id-{i}" for i in range(10_000) if audit_draw(1, f"id-{i}") < 0.5)
    called = {"n": 0}

    def get_stage2():
        called["n"] += 1
        return _pred("skip", 0.5, model="mid", pid=pid)

    d = decide_cascade(
        _pred("skip", 0.99, model="cheap", pid=pid), p, get_stage2=get_stage2
    )
    assert d.audit is True
    assert d.action == "escalate"
    assert d.reason == "audit"
    assert d.stage == 1
    assert called["n"] == 1


def test_audit_id_is_score_independent():
    p = _profile(audit_rate=0.5, audit_seed=7)
    pid = "case-42"
    assert audit_draw(7, pid) < 0.5 or audit_draw(7, pid) >= 0.5
    d1 = decide_cascade(
        _pred("skip", 0.99, model="cheap", pid=pid),
        p,
        stage2=_pred("skip", 0.99, model="mid", pid=pid),
    )
    d2 = decide_cascade(
        _pred("skip", 0.1, model="cheap", pid=pid),
        p,
        stage2=_pred("skip", 0.1, model="mid", pid=pid),
    )
    assert d1.audit == d2.audit


def test_load_save_round_trip_and_old_profile_still_loads(tmp_path):
    p = _profile()
    path = save_profile(p, tmp_path / "c.json")
    loaded = load_profile(path)
    assert isinstance(loaded, CascadeProfile)
    assert loaded.profile_hash == p.profile_hash
    assert loaded.stages[0].threshold == 0.9

    # Pre-kind single-model files keep loading.
    from decisionops.decide import CertificationProfile

    single = CertificationProfile(alpha=0.05, threshold=0.9, n_calibration=10)
    sp = save_profile(single, tmp_path / "s.json")
    raw = json.loads(sp.read_text())
    assert "kind" not in raw
    assert isinstance(load_profile(sp), CertificationProfile)


def test_unknown_kind_is_rejected(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps({"schema_version": 1, "kind": "triple", "alpha": 0.05, "threshold": 0.9})
        + "\n"
    )
    with pytest.raises(ValueError, match="unknown kind"):
        load_profile(path)


def test_cascade_profile_from_certificate():
    from decisionops.conformal.cascade_risk import StageArrays, certify_cascade

    n = 400
    import numpy as np

    rng = np.random.default_rng(1)
    c1 = rng.uniform(0.7, 0.99, n)
    c2 = rng.uniform(0.8, 0.99, n)
    s1 = StageArrays(c1, (rng.random(n) >= c1).astype(float), np.zeros(n))
    s2 = StageArrays(c2, (rng.random(n) >= c2).astype(float), np.zeros(n))
    cert = certify_cascade(
        s1, s2, 0.05, lambda1_grid=[0.85, 0.9], lambda2_grid=[0.85, 0.9], confidence_level=0.9
    )
    assert cert.feasible
    profile = cascade_profile_from(
        cert,
        CascadeStage(model="cheap", temperature=1.1),
        CascadeStage(model="mid", temperature=0.9),
        contract=ROUTING,
    )
    assert profile.stages[0].threshold == cert.lambda1
    assert profile.stages[1].temperature == 0.9
    assert profile.contract_hash == ROUTING.fingerprint()


def test_single_model_recertify_still_refuses_cascade():
    p = _profile()
    with pytest.raises(ValueError, match="run_cascade_recertification"):
        run_recertification(p, decisions=[], outcomes=[], predictions=[])  # type: ignore[arg-type]


def test_cli_decide_needs_two_prediction_files_for_cascade(tmp_path):
    path = save_profile(_profile(), tmp_path / "c.json")
    preds = tmp_path / "p.jsonl"
    preds.write_text(
        json.dumps(
            {
                "id": "x",
                "prediction": "skip",
                "probabilities": {"skip": 0.95, "deep": 0.05},
                "model": "cheap",
            }
        )
        + "\n"
    )
    result = runner.invoke(app, ["decide", str(path), "-p", str(preds)])
    assert result.exit_code != 0
    assert "cascade" in result.output.lower()
