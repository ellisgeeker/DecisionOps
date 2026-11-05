"""Label-free drift monitor."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract
from decisionops.decide import decide_many, profile_from_report, save_profile
from decisionops.drift import (
    population_stability_index,
    run_drift_monitor,
    window_stats,
)
from decisionops.schemas import DecisionCase, DecisionPrediction

runner = CliRunner()

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
)


def _preds(n: int, seed: int, *, conf_shift: float = 0.0, floor: float = 0.55):
    rng = np.random.default_rng(seed)
    cases, preds = [], []
    for i in range(n):
        conf = float(np.clip(rng.uniform(floor, 0.999) + conf_shift, 0.5, 0.999))
        p_ok = conf * conf
        label = "skip" if rng.random() < p_ok else "deep"
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


def _profile_decisions(n=600, seed=1, audit_rate=0.1, conf_shift=0.0):
    cases, preds = _preds(n, seed, conf_shift=conf_shift)
    report = run_certification(
        cases, preds, alphas=[0.10], contract=ROUTING, calibrate=False, seed=3
    )
    profile = profile_from_report(report, 0.10, audit_rate=audit_rate, audit_seed=7)
    profile = profile.model_copy(update={"model": "fixture-v1"})
    decisions = decide_many(preds, profile)
    return profile, decisions, preds


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(
            json.dumps(r if isinstance(r, dict) else r.model_dump(mode="json"))
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_psi_zero_on_identical_samples():
    x = np.linspace(0.5, 0.99, 200)
    assert population_stability_index(x, x.copy()) == 0.0


def test_psi_rises_when_mass_shifts():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 0.8, size=2000)
    live = rng.uniform(0.75, 0.99, size=2000)
    assert population_stability_index(base, live) > 0.25


def test_window_stats_certified_rate_ignores_audit_draw():
    profile, decisions, _ = _profile_decisions(n=500, audit_rate=0.4, seed=2)
    stats = window_stats(decisions)
    # Certified rate should track coverage much more closely than auto_rate when
    # audit_rate is large.
    assert stats.audit_rate_observed > 0.2
    assert stats.auto_rate < stats.certified_rate
    if profile.coverage_calibration is not None:
        assert abs(stats.certified_rate - profile.coverage_calibration) < 0.15


def test_matched_windows_are_ok():
    profile, base, _ = _profile_decisions(n=800, seed=10, audit_rate=0.1)
    # Fresh draw, same distribution, same profile.
    _, live, _ = _profile_decisions(n=800, seed=11, audit_rate=0.1)
    # Re-route live preds with the *same* profile object from the first call.
    cases, preds = _preds(800, 11)
    live = decide_many(preds, profile)
    report = run_drift_monitor(live, base, profile=profile, min_window=50)
    assert report.verdict == "ok"
    assert abs(report.delta_certified_rate) < 0.05
    assert report.psi_confidence is not None
    assert report.psi_confidence < 0.1


def test_confidence_shift_triggers_recertify_suggested():
    profile, base, _ = _profile_decisions(n=1000, seed=20, audit_rate=0.05, conf_shift=0.0)
    # Live traffic: confidences pushed up → more certified mass, different shape.
    cases, preds = _preds(1000, 21, conf_shift=0.25)
    live = decide_many(preds, profile)
    report = run_drift_monitor(live, base, profile=profile, min_window=50)
    assert report.verdict == "recertify_suggested"
    assert report.psi_confidence is not None
    assert report.psi_confidence >= 0.10 or abs(report.delta_certified_rate) >= 0.05
    assert report.live.mean_confidence > report.baseline.mean_confidence


def test_cli_drift_exit_codes(tmp_path):
    profile, base, _ = _profile_decisions(n=700, seed=30, audit_rate=0.08)
    profile_path = save_profile(profile, tmp_path / "p.json")
    base_path = tmp_path / "base.jsonl"
    live_ok = tmp_path / "live_ok.jsonl"
    live_bad = tmp_path / "live_bad.jsonl"
    _write_jsonl(base_path, base)

    _, preds_ok = _preds(700, 31)
    _write_jsonl(live_ok, decide_many(preds_ok, profile))
    _, preds_bad = _preds(700, 32, conf_shift=0.3)
    _write_jsonl(live_bad, decide_many(preds_bad, profile))

    out = tmp_path / "drift.json"
    ok = runner.invoke(
        app,
        [
            "drift",
            str(live_ok),
            "--baseline",
            str(base_path),
            "--profile",
            str(profile_path),
            "-o",
            str(out),
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "verdict=" in ok.output
    assert out.exists()

    bad = runner.invoke(
        app,
        [
            "drift",
            str(live_bad),
            "--baseline",
            str(base_path),
            "--profile",
            str(profile_path),
        ],
    )
    assert bad.exit_code == 2, bad.output
    assert "recertify_suggested" in bad.output


def test_rejects_tiny_windows():
    profile, base, _ = _profile_decisions(n=80, seed=40)
    try:
        run_drift_monitor(base[:5], base, profile=profile, min_window=30)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "at least 30" in str(exc)


# --- Cascade windows --------------------------------------------------------


def _cascade_pool(n: int, seed: int, *, conf_shift: float = 0.0, stage1_lo: float = 0.55):
    rng = np.random.default_rng(seed)
    cases, p1, p2 = [], [], []
    for i in range(n):
        cid = str(i)
        c1 = float(np.clip(rng.uniform(stage1_lo, 0.999) + conf_shift, 0.5, 0.999))
        c2 = float(np.clip(rng.uniform(0.70, 0.999) + conf_shift * 0.5, 0.5, 0.999))
        right = rng.random() < max(c1, c2) ** 2
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


def _cascade_profile_and_base(n=1200, seed=1, audit_rate=0.1):
    from decisionops.conformal.cascade_certify import run_cascade_certification
    from decisionops.decide import cascade_profile_from_report, decide_cascade_many

    cases, p1, p2 = _cascade_pool(n, seed)
    report = run_cascade_certification(
        cases, p1, p2, ROUTING, alpha=0.10, confidence_level=0.9, calibrate=False, seed=3
    )
    assert report.certificate.feasible
    profile = cascade_profile_from_report(report, audit_rate=audit_rate, audit_seed=7)
    decisions = decide_cascade_many(p1, profile, p2)
    return profile, decisions


def test_cascade_window_reports_stage_shares():

    profile, base = _cascade_profile_and_base(n=800, seed=40)
    stats = window_stats(base)
    assert stats.cascade is True
    assert stats.stage1_rate is not None and stats.stage2_rate is not None
    assert 0.0 <= stats.stage1_rate <= 1.0
    assert 0.0 <= stats.stage2_rate <= 1.0
    # Certified ≈ stage1 + stage2 (audit rows keep their stage).
    assert abs(stats.certified_rate - (stats.stage1_rate + stats.stage2_rate)) < 1e-9


def test_cascade_matched_windows_are_ok():
    from decisionops.decide import decide_cascade_many

    profile, base = _cascade_profile_and_base(n=1000, seed=50)
    _, p1, p2 = _cascade_pool(1000, 51)
    live = decide_cascade_many(p1, profile, p2)
    report = run_drift_monitor(live, base, profile=profile, min_window=50)
    assert report.verdict == "ok"
    assert report.delta_stage1_rate is not None
    assert abs(report.delta_certified_rate) < 0.05
    assert abs(report.delta_stage1_rate) < 0.05
    assert abs(report.delta_stage2_rate) < 0.05


def test_cascade_stage_share_shift_alerts():
    from decisionops.decide import CascadeStage, decide_cascade_many

    profile, base = _cascade_profile_and_base(n=1200, seed=60)
    # Force almost everything onto stage 2 by disabling stage 1 on a copy used
    # only to *produce* live decisions — then compare those decisions to base.
    # Real drift is traffic shift; here we simulate a routing-mix change.
    live_profile = profile.model_copy(
        update={
            "stages": [
                CascadeStage(
                    provider=profile.stages[0].provider,
                    model=profile.stages[0].model,
                    temperature=profile.stages[0].temperature,
                    threshold=0.999,  # almost never stage 1
                ),
                profile.stages[1],
            ]
        }
    )
    _, p1, p2 = _cascade_pool(1200, 61)
    live = decide_cascade_many(p1, live_profile, p2)
    # Compare without profile hash check (hashes differ by construction).
    report = run_drift_monitor(live, base, profile=None, min_window=50)
    assert report.baseline.cascade and report.live.cascade
    assert report.delta_stage1_rate is not None
    assert report.delta_stage1_rate < -0.05
    assert report.verdict in {"watch", "recertify_suggested"}


def test_cascade_rejects_mixed_with_single_model():
    profile, cascade_dec = _cascade_profile_and_base(n=200, seed=70)
    _, single_dec, _ = _profile_decisions(n=200, seed=71)
    try:
        run_drift_monitor(cascade_dec, single_dec, min_window=50)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "cascade" in str(exc).lower()


def test_cli_drift_accepts_cascade_profile(tmp_path):
    from decisionops.decide import decide_cascade_many, save_profile

    profile, base = _cascade_profile_and_base(n=700, seed=80)
    _, p1, p2 = _cascade_pool(700, 81)
    live = decide_cascade_many(p1, profile, p2)
    profile_path = save_profile(profile, tmp_path / "c.json")
    base_path = tmp_path / "base.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_jsonl(base_path, base)
    _write_jsonl(live_path, live)
    out = tmp_path / "drift.json"
    result = runner.invoke(
        app,
        [
            "drift",
            str(live_path),
            "--baseline",
            str(base_path),
            "--profile",
            str(profile_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "stage1_share" in result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["cascade"] is True
    assert payload["delta_stage1_rate"] is not None


def test_baseline_from_another_profile_is_flagged():
    """After a profile swap, an old baseline must be called out, not silently compared."""
    profile, base, _ = _profile_decisions(n=600, seed=90, audit_rate=0.1)
    newer = profile.model_copy(update={"threshold": min(0.999, profile.threshold + 0.05)})
    _, preds = _preds(600, 91)
    live = decide_many(preds, newer)
    report = run_drift_monitor(live, base, profile=newer, min_window=50)
    assert any(n.startswith("Baseline decisions reference profile") for n in report.notes)
    assert not any(n.startswith("Live decisions reference profile") for n in report.notes)
