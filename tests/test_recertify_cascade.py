"""Audit-only recertify for two-stage cascade profiles."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal.cascade_certify import run_cascade_certification
from decisionops.conformal.cascade_risk import (
    CascadeCertificate,
    CascadePoint,
    stage_arrays_from_pairs,
)
from decisionops.contract import DecisionContract
from decisionops.decide import (
    CascadeProfile,
    CascadeStage,
    Decision,
    cascade_profile_from_report,
    decide_cascade_many,
    load_profile,
    save_profile,
)
from decisionops.io import align_cases_and_predictions
from decisionops.recertify import (
    _cascade_verdict,
    build_cascade_audit_pairs,
    run_cascade_recertification,
    run_recertification,
)
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
)


def _pool(n: int, seed: int, *, drift: float = 0.0, prefix: str = ""):
    """
    Two stages that both predict `skip`; label is `skip` when the stage that
    ends up deciding would be right. `drift` lowers P(correct) at every
    confidence, so the same scores become over-confident.
    """
    rng = np.random.default_rng(seed)
    cases, p1, p2 = [], [], []
    for i in range(n):
        cid = f"{prefix}{i}"
        c1 = float(rng.uniform(0.55, 0.999))
        c2 = float(rng.uniform(0.70, 0.999))
        # One latent truth per case; each stage's confidence tracks it.
        right = rng.random() < max(0.0, max(c1, c2) ** 2 * (1.0 - drift))
        label = "skip" if right else "deep"
        cases.append(DecisionCase(id=cid, choices=["skip", "deep"], label=label))
        p1.append(
            DecisionPrediction(
                id=cid,
                prediction="skip",
                probabilities={"skip": c1, "deep": 1.0 - c1},
                model="cheap",
            )
        )
        p2.append(
            DecisionPrediction(
                id=cid,
                prediction="skip",
                probabilities={"skip": c2, "deep": 1.0 - c2},
                model="mid",
            )
        )
    return cases, p1, p2


def _live_profile(*, audit_rate: float = 0.5, n: int = 2000, seed: int = 0) -> CascadeProfile:
    cases, p1, p2 = _pool(n, seed)
    report = run_cascade_certification(
        cases, p1, p2, ROUTING, alpha=0.10, confidence_level=0.9, calibrate=False, seed=3
    )
    assert report.certificate.feasible
    return cascade_profile_from_report(report, audit_rate=audit_rate, audit_seed=5)


def _traffic(profile: CascadeProfile, *, n: int, seed: int, drift: float = 0.0):
    cases, p1, p2 = _pool(n, seed, drift=drift, prefix="live")
    decisions = decide_cascade_many(p1, profile, p2)
    outcomes = [OutcomeFeedback(id=c.id, label=c.label) for c in cases]
    return cases, p1, p2, decisions, outcomes


def _true_joint_error(profile: CascadeProfile, *, drift: float) -> float:
    """Joint automated error of the profile's point on a large fresh pool."""
    cases, p1, p2 = _pool(40_000, 999, drift=drift, prefix="truth")
    pairs1 = align_cases_and_predictions(cases, p1)
    pairs2 = align_cases_and_predictions(cases, p2)
    s1 = stage_arrays_from_pairs(pairs1, auto_choices=ROUTING.auto_set)
    s2 = stage_arrays_from_pairs(pairs2, auto_choices=ROUTING.auto_set)
    t1, t2 = (s.threshold for s in profile.stages)
    acc1 = np.zeros(len(cases), bool) if t1 is None else s1.confidence >= t1
    acc2 = ~acc1 & (np.zeros(len(cases), bool) if t2 is None else s2.confidence >= t2)
    return float((s1.wrong[acc1].sum() + s2.wrong[acc2].sum()) / len(cases))


def test_profile_stores_grids_for_recertify():
    profile = _live_profile()
    assert profile.lambda1_grid and profile.lambda2_grid
    if profile.stages[0].threshold is not None:
        assert profile.stages[0].threshold in profile.lambda1_grid


def test_matching_traffic_stays_feasible_and_writes_child_profile():
    profile = _live_profile()
    _, p1, p2, decisions, outcomes = _traffic(profile, n=1600, seed=1)
    report = run_cascade_recertification(
        profile, decisions=decisions, outcomes=outcomes, stage1=p1, stage2=p2, contract=ROUTING
    )
    assert report.verdict in {"keep", "replace"}
    assert report.certificate is not None and report.certificate.feasible
    assert report.n_audit_calibration == sum(d.audit for d in decisions)
    child = report.profile
    assert isinstance(child, CascadeProfile)
    assert child.split["protocol"] == "audit_recertify_cascade"
    assert child.split["parent_profile"] == profile.profile_hash
    # Frozen calibrators and audit settings carry over unchanged.
    assert [s.temperature for s in child.stages] == [s.temperature for s in profile.stages]
    assert (child.audit_rate, child.audit_seed) == (profile.audit_rate, profile.audit_seed)
    assert child.contract_hash == profile.contract_hash


def test_drift_makes_recertify_tighten_or_fail():
    profile = _live_profile()
    drift = 0.12
    old_risk = _true_joint_error(profile, drift=drift)
    assert old_risk > profile.alpha, "fixture drift must break the live certificate"

    _, p1, p2, decisions, outcomes = _traffic(profile, n=3000, seed=2, drift=drift)
    report = run_cascade_recertification(
        profile, decisions=decisions, outcomes=outcomes, stage1=p1, stage2=p2
    )
    # A stricter point must never be reported as "keep".
    assert report.verdict in {"replace", "infeasible"}
    if report.verdict == "replace":
        assert report.profile is not None
        assert _true_joint_error(report.profile, drift=drift) <= profile.alpha


def test_only_audit_rows_with_both_stages_enter():
    profile = _live_profile()
    cases, p1, p2, decisions, outcomes = _traffic(profile, n=400, seed=4)
    audit_ids = sorted(d.id for d in decisions if d.audit)
    dropped = audit_ids[0]
    p2_missing = [p for p in p2 if p.id != dropped]
    pairs1, pairs2, join = build_cascade_audit_pairs(
        decisions=decisions, outcomes=outcomes, stage1=p1, stage2=p2_missing, profile=profile
    )
    assert [c.id for c, _ in pairs1] == [c.id for c, _ in pairs2]
    assert {c.id for c, _ in pairs1} == set(audit_ids) - {dropped}
    assert join["n_skipped_missing_prediction"] == 1
    assert join["n_organic_labeled_ignored"] == len(decisions) - len(audit_ids)


def test_stage_model_mismatch_is_refused():
    profile = _live_profile()
    _, p1, p2, decisions, outcomes = _traffic(profile, n=200, seed=5)
    wrong = [p.model_copy(update={"model": "other"}) for p in p2]
    with pytest.raises(ValueError, match="cascade stage 2"):
        run_cascade_recertification(
            profile, decisions=decisions, outcomes=outcomes, stage1=p1, stage2=wrong
        )


def test_no_labels_is_infeasible_without_certificate():
    profile = _live_profile()
    _, p1, p2, decisions, _ = _traffic(profile, n=200, seed=6)
    report = run_cascade_recertification(
        profile, decisions=decisions, outcomes=[], stage1=p1, stage2=p2
    )
    assert report.verdict == "infeasible"
    assert report.certificate is None and report.profile is None


def test_single_model_entry_point_still_refuses_cascade():
    with pytest.raises(ValueError, match="run_cascade_recertification"):
        run_recertification(_live_profile(), decisions=[], outcomes=[], predictions=[])


def _cert(l1, l2, automation=0.5, feasible=True) -> CascadeCertificate:
    point = CascadePoint(
        lambda1=l1,
        lambda2=l2,
        n_total=100,
        n_auto_stage1=0,
        n_auto_stage2=0,
        n_errors=0,
        joint_error_rate=0.0,
        automation_rate=automation,
        stage2_call_rate=0.0,
    )
    return CascadeCertificate(
        alpha=0.05,
        confidence_level=0.9,
        feasible=feasible,
        lambda1=l1,
        lambda2=l2,
        objective="automation",
        n_calibration=100,
        n_rows=3,
        per_row_level=0.96,
        min_n_required=60,
        calibration=point if feasible else None,
    )


def test_verdict_replaces_on_any_move_including_stricter():
    live = CascadeProfile(
        alpha=0.05,
        confidence_level=0.9,
        stages=[CascadeStage(threshold=0.9), CascadeStage(threshold=0.8)],
        n_calibration=100,
        coverage_calibration=0.5,
    )
    assert _cascade_verdict(live, _cert(0.9, 0.8)) == "keep"
    assert _cascade_verdict(live, _cert(0.95, 0.8)) == "replace"  # stricter
    assert _cascade_verdict(live, _cert(0.9, None)) == "replace"  # stage 2 switched off
    assert _cascade_verdict(live, _cert(0.9, 0.8, automation=0.4)) == "replace"
    assert _cascade_verdict(live, _cert(None, None, feasible=False)) == "infeasible"


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r.model_dump(mode="json")) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_cli_recertify_cascade_round_trip(tmp_path):
    profile = _live_profile()
    _, p1, p2, decisions, outcomes = _traffic(profile, n=1600, seed=7)
    live = save_profile(profile, tmp_path / "live.json")
    paths = {}
    for name, rows in {
        "decisions": decisions,
        "outcomes": outcomes,
        "cheap": p1,
        "mid": p2,
    }.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], rows)
    out_profile = tmp_path / "next.json"
    out_report = tmp_path / "recert.json"
    result = runner.invoke(
        app,
        [
            "recertify",
            str(live),
            "--decisions",
            str(paths["decisions"]),
            "--outcomes",
            str(paths["outcomes"]),
            "-p",
            f"cheap={paths['cheap']}",
            "-p",
            f"mid={paths['mid']}",
            "--profile-out",
            str(out_profile),
            "-o",
            str(out_report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cascade" in result.output
    child = load_profile(out_profile)
    assert isinstance(child, CascadeProfile)
    payload = json.loads(out_report.read_text(encoding="utf-8"))
    assert payload["verdict"] in {"keep", "replace"}
    assert payload["certificate"]["feasible"] is True


def test_cli_recertify_cascade_needs_two_predictions(tmp_path):
    profile = _live_profile()
    _, p1, _, decisions, outcomes = _traffic(profile, n=100, seed=8)
    live = save_profile(profile, tmp_path / "live.json")
    d, o, c = (tmp_path / f"{n}.jsonl" for n in ("d", "o", "c"))
    _write_jsonl(d, decisions)
    _write_jsonl(o, outcomes)
    _write_jsonl(c, p1)
    result = runner.invoke(
        app, ["recertify", str(live), "--decisions", str(d), "--outcomes", str(o), "-p", str(c)]
    )
    assert result.exit_code != 0
    assert "two" in result.output.lower()


def test_decision_audit_rows_are_score_independent_for_cascade():
    """Guard the premise: the audit set is a uniform sample, not tied to stage."""
    profile = _live_profile(audit_rate=0.3)
    _, _, _, decisions, _ = _traffic(profile, n=4000, seed=9)
    audited = [d for d in decisions if d.audit]
    rate_auto_path = np.mean([d.stage is not None for d in audited])
    rate_all = np.mean([d.stage is not None for d in decisions])
    assert abs(rate_auto_path - rate_all) < 0.04
    assert all(isinstance(d, Decision) for d in audited)
