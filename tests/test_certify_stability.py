"""Error budget and re-split stability on certificates."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from decisionops.cli.main import app
from decisionops.conformal.certify import run_certification
from decisionops.conformal.risk_control import (
    certify_threshold,
    clopper_pearson_upper,
    hp_error_budget,
)
from decisionops.schemas import DecisionCase, DecisionPrediction


def _pool(n: int, error_rate: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    cases, preds = [], []
    for i in range(n):
        conf = float(rng.uniform(0.55, 0.99))
        wrong = rng.random() < error_rate * (1.6 - conf)  # less confident -> more errors
        label = "b" if wrong else "a"
        cases.append(DecisionCase(id=str(i), choices=["a", "b"], label=label))
        preds.append(
            DecisionPrediction(id=str(i), prediction="a", probabilities={"a": conf, "b": 1 - conf})
        )
    return cases, preds


def test_crc_error_budget_is_alpha_n_plus_1_minus_b():
    cases, preds = _pool(400, 0.05)
    pairs = list(zip(cases, preds, strict=True))
    cert = certify_threshold(pairs, 0.05)
    assert cert.error_budget == pytest.approx(0.05 * 401 - 1.0)
    assert cert.n_errors_calibration == sum(c.label != "a" for c in cases)
    weighted = certify_threshold(pairs, 0.05, loss_bound=5.0)
    assert weighted.error_budget == pytest.approx(0.05 * 401 - 5.0)


def test_hp_error_budget_is_largest_passing_k():
    n, alpha, cl = 500, 0.05, 0.9
    k = hp_error_budget(n, alpha, cl)
    assert clopper_pearson_upper(k, n, cl) <= alpha
    assert clopper_pearson_upper(k + 1, n, cl) > alpha
    cases, preds = _pool(n, 0.05)
    cert = certify_threshold(list(zip(cases, preds, strict=True)), alpha, confidence_level=cl)
    assert cert.error_budget == k
    # Below the HP floor no error count passes.
    assert hp_error_budget(10, 0.05, 0.9) == -1


def test_few_errors_note_only_when_budget_is_small():
    # n_cal = 0.3 * 1000 = 300 at alpha=1%: budget 0.01*301 - 1 = 2.01, feasible.
    cases, preds = _pool(1000, 0.002)
    small = run_certification(cases, preds, alphas=[0.01], calibrate=False)
    cert = small.levels[0].certificate
    assert cert.feasible and cert.error_budget == pytest.approx(2.01)
    assert any("Few errors carry the certificate" in n for n in small.notes)
    # Infeasible levels do not get the note: there is no certificate to doubt.
    tiny = run_certification(cases[:200], preds[:200], alphas=[0.01], calibrate=False)
    assert not tiny.levels[0].certificate.feasible
    assert not any("Few errors" in n for n in tiny.notes)

    cases, preds = _pool(3000, 0.1)
    big = run_certification(cases, preds, alphas=[0.10], calibrate=False)  # budget ~89
    assert big.levels[0].certificate.error_budget > 5
    assert not any("Few errors" in n for n in big.notes)


def test_stability_splits_report_spread_and_flag_wide_coverage():
    cases, preds = _pool(1200, 0.03, seed=1)
    report = run_certification(
        cases, preds, alphas=[0.02, 0.05], calibrate=False, stability_splits=12
    )
    for level in report.levels:
        s = level.stability
        assert s is not None and s.n_splits == 12
        assert 0 < s.n_feasible <= 12
        for spread in (s.threshold, s.test_coverage, s.test_joint_error):
            assert spread["min"] <= spread["p10"] <= spread["median"] <= spread["p90"]
            assert spread["p90"] <= spread["max"]
        assert s.mean_test_joint_error is not None
    assert report.metadata["stability_splits"] == 12
    # The reruns use other seeds: the primary certificate is unchanged.
    plain = run_certification(cases, preds, alphas=[0.02, 0.05], calibrate=False)
    for a, b in zip(report.levels, plain.levels, strict=True):
        assert a.certificate.threshold == b.certificate.threshold


def test_stability_is_off_by_default_and_rejects_negative():
    cases, preds = _pool(300, 0.05)
    report = run_certification(cases, preds, alphas=[0.1], calibrate=False)
    assert report.levels[0].stability is None
    with pytest.raises(ValueError, match="stability_splits"):
        run_certification(cases, preds, alphas=[0.1], calibrate=False, stability_splits=-1)


def test_wide_coverage_note_fires_on_a_noisy_certificate():
    """
    Few errors and a threshold that binds (base error rate near alpha), as on
    cpython: one certificate moves a lot from split to split.
    """
    cases, preds = _pool(1000, 0.025, seed=3)
    report = run_certification(
        cases, preds, alphas=[0.01], calibrate=False, stability_splits=20
    )
    cov = report.levels[0].stability.test_coverage
    assert cov["p90"] - cov["p10"] > 0.2
    assert any("certified test coverage moves" in n for n in report.notes)


def test_wide_coverage_note_silent_when_stable():
    cases, preds = _pool(3000, 0.1, seed=4)
    report = run_certification(
        cases, preds, alphas=[0.10], calibrate=False, stability_splits=10
    )
    cov = report.levels[0].stability.test_coverage
    assert cov["p90"] - cov["p10"] <= 0.2
    assert not any("certified test coverage moves" in n for n in report.notes)


def test_cli_stability_splits(tmp_path):
    cases, preds = _pool(600, 0.05)
    ds, pr = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    ds.write_text("".join(json.dumps(c.model_dump()) + "\n" for c in cases))
    pr.write_text("".join(json.dumps(p.model_dump()) + "\n" for p in preds))
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        ["certify", str(ds), "-p", str(pr), "--alphas", "0.1", "--no-calibrate",
         "--stability-splits", "5", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "Stability over 5 re-splits" in text
    assert "Error budget" in text
    saved = json.loads((out / "certification_report.json").read_text())
    assert saved["levels"][0]["stability"]["n_splits"] == 5
    bad = CliRunner().invoke(
        app, ["certify", str(ds), "-p", str(pr), "--stability-splits", "-1"]
    )
    assert bad.exit_code != 0
