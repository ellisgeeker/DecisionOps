"""Compare several models on one dataset under an identical gate + CRC protocol."""

from __future__ import annotations

from pathlib import Path

from decisionops.conformal.certify import CertificationReport, run_certification
from decisionops.contract import DecisionContract
from decisionops.eval import run_gate
from decisionops.io import load_dataset, load_predictions
from decisionops.schemas import (
    ComparisonEntry,
    ComparisonReport,
    DecisionCase,
    DecisionPrediction,
    GateReport,
)


def _risk_key(risk: float) -> str:
    return f"{risk:.4f}".rstrip("0").rstrip(".")


def _certified_fields(cert: CertificationReport) -> dict:
    feasible: dict[str, bool] = {}
    threshold: dict[str, float] = {}
    coverage: dict[str, float] = {}
    joint: dict[str, float] = {}
    for level in cert.levels:
        key = _risk_key(level.certificate.alpha)
        feasible[key] = bool(level.certificate.feasible)
        if not level.certificate.feasible or level.test is None:
            continue
        if level.certificate.threshold is not None:
            threshold[key] = float(level.certificate.threshold)
        coverage[key] = float(level.test["coverage"])
        joint[key] = float(level.test["joint_error_rate"])
    return {
        "n_crc_calibration": cert.split.n_calibration,
        "n_temperature_fit": cert.n_temperature_fit,
        "certified_feasible_by_risk": feasible,
        "certified_threshold_by_risk": threshold,
        "certified_coverage_by_risk": coverage,
        "certified_joint_error_by_risk": joint,
    }


def _entry_from_gate(
    name: str,
    report: GateReport,
    cert: CertificationReport,
) -> ComparisonEntry:
    raw = next(c for c in report.conditions if c.name == "raw")
    cal = next((c for c in report.conditions if c.name == "temperature_scaled"), None)
    focus = cal or raw

    coverage: dict[str, float] = {}
    sel_risk: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for t in focus.thresholds:
        key = _risk_key(t.target_risk)
        coverage[key] = t.coverage
        sel_risk[key] = t.selective_risk
        thresholds[key] = t.threshold

    return ComparisonEntry(
        name=name,
        provider=report.provider,
        model=report.model,
        n_test=report.split.n_test,
        accuracy=focus.accuracy,
        ece_raw=raw.ece,
        ece_calibrated=cal.ece if cal else None,
        brier_raw=raw.brier,
        brier_calibrated=cal.brier if cal else None,
        nll_raw=raw.nll,
        nll_calibrated=cal.nll if cal else None,
        temperature=report.temperature,
        coverage_by_risk=coverage,
        selective_risk_by_risk=sel_risk,
        threshold_by_risk=thresholds,
        latency_p95_ms=report.latency.p95_ms,
        verdict=report.verdict.recommendation,
        **_certified_fields(cert),
    )


def compare_models(
    cases: list[DecisionCase],
    prediction_sets: dict[str, list[DecisionPrediction]],
    *,
    dataset_name: str = "dataset",
    cal_ratio: float = 0.3,
    seed: int = 42,
    n_bins: int = 10,
    target_risks: list[float] | None = None,
    calibrate: bool = True,
    temp_ratio: float = 0.2,
    contract: DecisionContract | None = None,
) -> ComparisonReport:
    """
    Run gate + CRC on every prediction set under one shared split seed.

    Rankings use CRC feasibility (decided on the CRC calibration set) and
    certified test coverage. Empirical selective-risk figures stay on the entry
    for contrast; they do not pick the winner. A single test split landing
    above alpha does not disqualify a certificate — the guarantee is on E[loss].
    """
    if not prediction_sets:
        raise ValueError("need at least one prediction set to compare")
    if target_risks is None:
        target_risks = [0.01, 0.05, 0.10]
    if not 0.0 <= temp_ratio < 1.0:
        raise ValueError(f"temp_ratio must be in [0, 1), got {temp_ratio}")

    entries: list[ComparisonEntry] = []
    split_info = None
    n_crc_cal = None
    for name, preds in prediction_sets.items():
        report = run_gate(
            cases,
            preds,
            cal_ratio=cal_ratio,
            seed=seed,
            n_bins=n_bins,
            target_risks=target_risks,
            calibrate=calibrate,
            contract=contract,
            temp_ratio=temp_ratio,
        )
        cert = run_certification(
            cases,
            preds,
            alphas=list(target_risks),
            cal_ratio=cal_ratio,
            temp_ratio=temp_ratio if calibrate else 0.0,
            seed=seed,
            calibrate=calibrate,
            contract=contract,
        )
        split_info = report.split
        n_crc_cal = cert.split.n_calibration
        entries.append(_entry_from_gate(name, report, cert))

    assert split_info is not None

    best: dict[str, str] = {}
    feasible_counts: dict[str, int] = {}
    notes: list[str] = [
        "Ranking uses CRC certificates (joint automated error rate), not empirical "
        "selective risk. Empirical figures stay in the report for contrast and are "
        "marked ! when they rebound on test.",
        "CRC certifies E[loss] <= alpha under exchangeability. A single test split "
        "landing above alpha is expected roughly half the time and does not void "
        "the certificate.",
    ]
    for risk in target_risks:
        key = _risk_key(risk)
        feasible = [
            e
            for e in entries
            if e.certified_feasible_by_risk.get(key)
            and key in e.certified_coverage_by_risk
        ]
        feasible_counts[key] = len(feasible)

        if feasible:
            winner = max(feasible, key=lambda e: e.certified_coverage_by_risk[key])
            best[key] = winner.name
            n_infeas = len(entries) - len(feasible)
            if n_infeas:
                names = [
                    e.name
                    for e in entries
                    if not e.certified_feasible_by_risk.get(key)
                ]
                notes.append(
                    f"@{risk:.0%}: {n_infeas} of {len(entries)} infeasible "
                    f"({', '.join(names)}) — no certificate, excluded from ranking."
                )
            if winner.certified_coverage_by_risk[key] < 0.10:
                notes.append(
                    f"@{risk:.0%}: winner {winner.name} only covers "
                    f"{winner.certified_coverage_by_risk[key]:.1%} — holding the "
                    f"budget here costs almost all automation."
                )
        else:
            notes.append(
                f"@{risk:.0%}: no model has a feasible CRC certificate. "
                "Abstain on all traffic at this level."
            )

    return ComparisonReport(
        dataset=dataset_name,
        split=split_info,
        target_risks=list(target_risks),
        entries=entries,
        best_by_coverage_at=best,
        notes=notes,
        metadata={
            "protocol": "shared_split_gate_and_crc",
            "seed": seed,
            "temp_ratio": temp_ratio,
            "n_crc_calibration": n_crc_cal,
            "ranking_rule": (
                "max certified test coverage among models with a feasible "
                "CRC certificate (joint automated error rate)"
            ),
            "n_feasible_by_risk": feasible_counts,
            "certified_quantity": "joint_automated_error_rate",
            "contract": contract.summary() if contract is not None else None,
        },
    )


def load_prediction_sets(specs: list[str]) -> dict[str, list[DecisionPrediction]]:
    """
    Parse `name=path` (or bare `path`) specs into prediction sets.
    """
    out: dict[str, list[DecisionPrediction]] = {}
    for spec in specs:
        if "=" in spec:
            name, _, raw_path = spec.partition("=")
            name = name.strip()
            path = Path(raw_path.strip())
        else:
            path = Path(spec)
            name = path.stem
        if not name:
            raise ValueError(f"invalid prediction spec: {spec!r}")
        if name in out:
            raise ValueError(f"duplicate prediction name: {name!r}")
        out[name] = load_predictions(path)
    return out


def load_cases(path: str | Path) -> list[DecisionCase]:
    return load_dataset(path)
