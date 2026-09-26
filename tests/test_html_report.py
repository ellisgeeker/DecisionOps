"""HTML report contains the two core charts from a gate run."""

from __future__ import annotations

from decisionops.eval import run_gate
from decisionops.report.html import write_html_report
from decisionops.schemas import DecisionCase, DecisionPrediction


def _pool(n: int = 80):
    cases, preds = [], []
    for i in range(n):
        correct = i % 5 != 0
        conf = 0.92 if correct else 0.61
        cases.append(
            DecisionCase(id=str(i), choices=["a", "b"], label="a" if correct else "b")
        )
        preds.append(
            DecisionPrediction(
                id=str(i),
                prediction="a",
                probabilities={"a": conf, "b": 1.0 - conf},
                provider="static",
                model="fixture",
            )
        )
    return cases, preds


def test_gate_html_contains_both_core_charts(tmp_path):
    cases, preds = _pool()
    report = run_gate(cases, preds, cal_ratio=0.3, seed=1, calibrate=True)
    focus = next(c for c in report.conditions if c.name == "temperature_scaled")
    assert focus.reliability_bins
    assert focus.risk_coverage
    assert any(int(p["n_accepted"]) > 0 for p in focus.risk_coverage)

    path = write_html_report(report, tmp_path / "gate_report.html")
    text = path.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "Reliability (raw)" in text
    assert "Reliability (temperature-scaled)" in text
    assert "Reliability (isotonic)" in text
    assert "Risk–coverage" in text
    assert "predicted confidence" in text
    assert "actual accuracy" in text
    assert "empirical" in text.lower()
    # No external assets — the report must open offline.
    assert "http://" not in text and "https://" not in text


def test_gate_html_empty_bins_do_not_crash(tmp_path):
    cases, preds = _pool(40)
    report = run_gate(cases, preds, cal_ratio=0.3, seed=2, n_bins=10, calibrate=False)
    path = write_html_report(report, tmp_path / "g.html")
    assert path.exists()
    assert "<svg" in path.read_text(encoding="utf-8")
