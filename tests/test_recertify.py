"""Outcome feedback + audit-only recertify."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract
from decisionops.decide import decide_many, load_profile, profile_from_report, save_profile
from decisionops.io import load_decisions, load_outcomes
from decisionops.recertify import build_audit_pairs, run_recertification
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
)


def _pool(n: int = 800, seed: int = 0, *, drift: float = 0.0):
    """
    Binary routing pool. `drift` multiplies P(correct | conf) downward so the
    same scores become over-confident — a stand-in for post-deploy shift.
    """
    rng = np.random.default_rng(seed)
    cases, preds = [], []
    for i in range(n):
        conf = float(rng.uniform(0.55, 0.999))
        p_correct = min(1.0, max(0.0, conf * conf * (1.0 - drift)))
        label = "skip" if rng.random() < p_correct else "deep"
        cases.append(DecisionCase(id=str(i), choices=["skip", "deep"], label=label))
        preds.append(
            DecisionPrediction(
                id=str(i),
                prediction="skip",
                probabilities={"skip": conf, "deep": 1.0 - conf},
                model="fixture-v1",
            )
        )
    return cases, preds


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r if isinstance(r, dict) else r.model_dump(mode="json")) for r in rows)
        + "\n",
        encoding="utf-8",
    )


def _profile_and_decisions(tmp_path, *, n=800, audit_rate=0.25, seed=1, drift=0.0):
    cases, preds = _pool(n=n, seed=seed, drift=drift)
    report = run_certification(
        cases, preds, alphas=[0.10], contract=ROUTING, calibrate=False, seed=7
    )
    profile = profile_from_report(report, 0.10, audit_rate=audit_rate, audit_seed=3)
    profile = profile.model_copy(update={"model": "fixture-v1"})
    decisions = decide_many(preds, profile)
    return cases, preds, profile, decisions


def test_outcome_feedback_round_trip(tmp_path):
    path = tmp_path / "out.jsonl"
    _write_jsonl(
        path,
        [
            OutcomeFeedback(id="1", label="deep", label_source="audit", matured_at="2026-01-01"),
            {"id": "2", "label": "skip"},
        ],
    )
    rows = load_outcomes(path)
    assert rows[0].label_source == "audit"
    assert rows[1].label_source == "audit"  # default
    assert rows[1].matured_at is None


def test_build_audit_pairs_ignores_organic_and_respects_maturity():
    cases, preds, profile, decisions = _profile_and_decisions(
        None, n=400, audit_rate=0.3, seed=2
    )
    # Label every case; mark half the non-audit as organic outcomes too.
    outcomes = []
    for c in cases:
        d = next(x for x in decisions if x.id == c.id)
        src = "audit" if d.audit else "escalation"
        matured = "2026-06-01" if int(c.id) % 2 == 0 else "2026-08-01"
        outcomes.append(
            OutcomeFeedback(id=c.id, label=c.label, label_source=src, matured_at=matured)
        )

    pairs, stats = build_audit_pairs(
        decisions=decisions,
        outcomes=outcomes,
        predictions=preds,
        profile=profile,
        as_of="2026-07-01",
    )
    assert stats["n_organic_labeled_ignored"] > 0
    assert stats["n_skipped_unmatured"] > 0
    assert all(d.audit for d in decisions if d.id in {c.id for c, _ in pairs})
    # Only matured_at <= as_of.
    assert stats["n_audit_labeled"] == len(pairs)
    assert len(pairs) < stats["n_audit_decisions"]


def test_recertify_rejects_non_audit_in_calibration_set():
    """Organic escalations must not move the certificate even if labeled."""
    cases, preds, profile, decisions = _profile_and_decisions(
        None, n=600, audit_rate=0.05, seed=4
    )
    # Label ONLY non-audit rows (pathological).
    outcomes = [
        OutcomeFeedback(id=c.id, label=c.label, label_source="escalation")
        for c, d in zip(cases, decisions, strict=True)
        if not d.audit
    ]
    report = run_recertification(
        profile, decisions=decisions, outcomes=outcomes, predictions=preds, contract=ROUTING
    )
    assert report.verdict == "infeasible"
    assert report.n_audit_calibration == 0
    assert report.join["n_organic_labeled_ignored"] == len(outcomes)


def test_recertify_on_matching_traffic_stays_feasible():
    cases, preds, profile, decisions = _profile_and_decisions(
        None, n=1200, audit_rate=0.4, seed=5
    )
    outcomes = [
        OutcomeFeedback(id=c.id, label=c.label, matured_at="2026-01-15")
        for c in cases
    ]
    report = run_recertification(
        profile,
        decisions=decisions,
        outcomes=outcomes,
        predictions=preds,
        contract=ROUTING,
        as_of="2026-02-01",
    )
    assert report.n_audit_calibration >= 100
    assert report.certificate.feasible
    assert report.verdict in {"keep", "replace"}
    assert report.profile is not None
    assert report.profile.temperature == profile.temperature
    assert report.profile.audit_rate == profile.audit_rate
    assert report.profile.split["protocol"] == "audit_recertify"
    assert report.profile.split["parent_profile"] == profile.profile_hash


def test_recertify_detects_drift_and_tightens_or_fails():
    """
    Certify on clean traffic, then label an audit sample drawn under drift.

    The drifted audit set should either raise the threshold or become
    infeasible relative to the original operating point.
    """
    # Live profile from clean pool.
    clean_cases, clean_preds, profile, _ = _profile_and_decisions(
        None, n=1000, audit_rate=0.5, seed=9, drift=0.0
    )
    assert profile.threshold is not None

    # New traffic under drift; same profile routes it (including audit draws).
    drift_cases, drift_preds = _pool(n=1000, seed=99, drift=0.55)
    # Keep model tag aligned.
    drift_preds = [p.model_copy(update={"model": "fixture-v1"}) for p in drift_preds]
    decisions = decide_many(drift_preds, profile)
    outcomes = [OutcomeFeedback(id=c.id, label=c.label) for c in drift_cases]

    report = run_recertification(
        profile,
        decisions=decisions,
        outcomes=outcomes,
        predictions=drift_preds,
        contract=ROUTING,
    )
    assert report.n_audit_calibration > 50
    if report.certificate.feasible:
        assert report.verdict == "replace"
        assert report.profile is not None
        # Stricter bar or lower coverage — drift should not relax automation.
        assert report.profile.threshold >= profile.threshold - 1e-9
    else:
        assert report.verdict == "infeasible"


def test_cli_recertify_round_trip(tmp_path):
    cases, preds, profile, decisions = _profile_and_decisions(
        tmp_path, n=900, audit_rate=0.35, seed=11
    )
    profile_path = save_profile(profile, tmp_path / "live.json")
    decisions_path = tmp_path / "decisions.jsonl"
    outcomes_path = tmp_path / "outcomes.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    contract_path = tmp_path / "contract.json"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(outcomes_path, [OutcomeFeedback(id=c.id, label=c.label) for c in cases])
    _write_jsonl(preds_path, preds)
    contract_path.write_text(ROUTING.model_dump_json(), encoding="utf-8")

    out_profile = tmp_path / "next.json"
    out_report = tmp_path / "recert.json"
    result = runner.invoke(
        app,
        [
            "recertify",
            str(profile_path),
            "--decisions",
            str(decisions_path),
            "--outcomes",
            str(outcomes_path),
            "-p",
            str(preds_path),
            "--contract",
            str(contract_path),
            "--profile-out",
            str(out_profile),
            "-o",
            str(out_report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recertify" in result.output
    assert out_profile.exists()
    loaded = load_profile(out_profile)
    assert loaded.alpha == profile.alpha
    assert loaded.split["protocol"] == "audit_recertify"
    payload = json.loads(out_report.read_text(encoding="utf-8"))
    assert payload["n_audit_calibration"] > 0
    assert "verdict" in payload

    # load_decisions works on the CLI artifact
    assert len(load_decisions(decisions_path)) == len(decisions)


def test_cli_recertify_infeasible_exits_nonzero(tmp_path):
    cases, preds, profile, decisions = _profile_and_decisions(
        tmp_path, n=200, audit_rate=0.02, seed=13
    )
    profile_path = save_profile(profile, tmp_path / "live.json")
    # Only a handful of audit labels → below CRC floor at tight alpha profiles;
    # force empty by providing no outcomes.
    decisions_path = tmp_path / "decisions.jsonl"
    outcomes_path = tmp_path / "outcomes.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(outcomes_path, [])  # empty file rejected by loader
    _write_jsonl(preds_path, preds)
    # empty outcomes file
    outcomes_path.write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "recertify",
            str(profile_path),
            "--decisions",
            str(decisions_path),
            "--outcomes",
            str(outcomes_path),
            "-p",
            str(preds_path),
            "--profile-out",
            str(tmp_path / "next.json"),
        ],
    )
    assert result.exit_code != 0


def test_label_loop_restores_alpha_after_concept_drift():
    """
    The loop's core claim, with closed-form true risk (scripts/measure_label_loop.py).

    Same scores, accuracy down 10 points at every confidence: the old threshold
    sits far above alpha, and recertify on ~100 audit labels alone brings mean
    true risk back to alpha.

    Teeth: at 100 labels the CRC penalty B/(n+1) shifts mean risk by ~0.010
    (measured 0.0471 vs 0.0568 without it, se ~0.001 at 600 reps). With 200
    reps the se is ~0.0016, so the mean-vs-alpha check fails a penalty-free
    rule. At ~400 labels the gap shrinks to ~0.003 and such a rule would pass.
    """
    import importlib.util
    from pathlib import Path

    from decisionops.conformal.risk_control import certify_threshold

    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_label_loop.py"
    spec = importlib.util.spec_from_file_location("measure_label_loop", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    alpha, lo, delta, n_audit = 0.05, 0.55, 0.10, 100
    old, new = [], []
    for rep in range(200):
        rng = np.random.default_rng(81_000 + rep)
        cases, preds = mod.pool(rng, 600, lo, 0.0, f"pre{rep}")
        report = run_certification(
            cases, preds, alphas=[alpha], contract=ROUTING, calibrate=False, seed=rep
        )
        old.append(mod.true_risk(profile_from_report(report, alpha).threshold, lo, delta))
        # The audit sample is uniform over drifted traffic: draw it directly.
        cases, preds = mod.pool(rng, n_audit, lo, delta, f"audit{rep}")
        cert = certify_threshold(
            list(zip(cases, preds, strict=True)), alpha, auto_choices=ROUTING.auto_set
        )
        if cert.feasible:
            new.append(mod.true_risk(cert.threshold, lo, delta))

    # Drift really broke the live certificate (a few small pre-drift pools
    # certify a very strict threshold that still holds, hence mean, not min).
    assert np.mean(old) > alpha + 0.02
    assert np.mean(np.array(old) > alpha) > 0.9
    assert len(new) >= 190
    mean, se = float(np.mean(new)), float(np.std(new, ddof=1) / np.sqrt(len(new)))
    assert mean <= alpha + 1.0 * se, f"recertified mean risk {mean:.4f} (se {se:.4f})"
    assert mean > alpha - 0.012, f"recertify far too conservative: {mean:.4f}"
