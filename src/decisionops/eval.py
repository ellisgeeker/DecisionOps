"""End-to-end evaluation and technical-gate reports."""

from __future__ import annotations

from decisionops.calibration.isotonic import (
    apply_isotonic_to_pairs,
    fit_isotonic,
)
from decisionops.calibration.temperature import (
    apply_temperature_to_pairs,
    fit_temperature,
)
from decisionops.conformal.certify import CertifiedLevel, run_certification
from decisionops.contract import DecisionContract, auto_set_of
from decisionops.io import align_cases_and_predictions
from decisionops.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    latency_stats,
    negative_log_likelihood,
)
from decisionops.metrics.calibration import reliability_bins
from decisionops.schemas import (
    ConditionResult,
    DecisionCase,
    DecisionPrediction,
    EvalResult,
    GateReport,
    GateVerdict,
    SplitInfo,
    ThresholdRecommendation,
)
from decisionops.selective import (
    evaluate_fixed_thresholds,
    recommend_thresholds,
    risk_coverage_curve,
)
from decisionops.selective.threshold import RiskCoveragePoint
from decisionops.split import split_pairs


def _curve_payload(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    auto_choices: frozenset[str] | None = None,
) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for p in risk_coverage_curve(pairs, auto_choices=auto_choices):
        point: dict[str, float | int] = {
            "threshold": p.threshold,
            "coverage": p.coverage,
            "n_accepted": p.n_accepted,
            "n_total": p.n_total,
        }
        if p.n_accepted > 0:
            point["selective_accuracy"] = p.selective_accuracy
            point["selective_risk"] = p.selective_risk
        out.append(point)
    return out


def _pairs_metrics(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    *,
    n_bins: int,
    thresholds: list[ThresholdRecommendation] | None = None,
    cal_thresholds: list[ThresholdRecommendation] | None = None,
    name: str = "raw",
    temperature: float | None = None,
    auto_choices: frozenset[str] | None = None,
) -> ConditionResult:
    return ConditionResult(
        name=name,
        temperature=temperature,
        n_samples=len(pairs),
        accuracy=accuracy(pairs),
        ece=expected_calibration_error(pairs, n_bins=n_bins),
        brier=brier_score(pairs),
        nll=negative_log_likelihood(pairs),
        thresholds=thresholds or [],
        cal_thresholds=cal_thresholds or [],
        reliability_bins=[
            {
                "bin_low": b.bin_low,
                "bin_high": b.bin_high,
                "count": b.count,
                "avg_confidence": b.avg_confidence,
                "accuracy": b.accuracy,
                "gap": b.gap,
            }
            for b in reliability_bins(pairs, n_bins=n_bins)
        ],
        risk_coverage=_curve_payload(pairs, auto_choices),
    )


def evaluate(
    cases: list[DecisionCase],
    predictions: list[DecisionPrediction],
    *,
    n_bins: int = 10,
    target_risks: list[float] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> EvalResult:
    """
    Compute core reliability metrics.

    Prefer `run_gate` when you need proper cal/test separation.
    This helper keeps a simple same-set path for unit tests and smoke checks.
    """
    pairs = align_cases_and_predictions(cases, predictions)
    preds_only = [p for _, p in pairs]

    providers = {p.provider for p in preds_only if p.provider}
    models = {p.model for p in preds_only if p.model}
    inferred_provider = provider or (next(iter(providers)) if len(providers) == 1 else None)
    inferred_model = model or (next(iter(models)) if len(models) == 1 else None)

    return EvalResult(
        n_samples=len(pairs),
        accuracy=accuracy(pairs),
        ece=expected_calibration_error(pairs, n_bins=n_bins),
        brier=brier_score(pairs),
        nll=negative_log_likelihood(pairs),
        latency=latency_stats(preds_only),
        thresholds=recommend_thresholds(pairs, target_risks=target_risks),
        provider=inferred_provider,
        model=inferred_model,
        metadata={
            "n_bins": n_bins,
            "protocol": "same_set",
            "warning": "thresholds fit and evaluated on the same labeled set",
            "reliability_bins": [
                {
                    "bin_low": b.bin_low,
                    "bin_high": b.bin_high,
                    "count": b.count,
                    "avg_confidence": b.avg_confidence,
                    "accuracy": b.accuracy,
                    "gap": b.gap,
                }
                for b in reliability_bins(pairs, n_bins=n_bins)
            ],
            "risk_coverage": _curve_payload(pairs),
        },
    )


def risk_coverage(
    cases: list[DecisionCase],
    predictions: list[DecisionPrediction],
    thresholds: list[float] | None = None,
) -> list[RiskCoveragePoint]:
    pairs = align_cases_and_predictions(cases, predictions)
    return risk_coverage_curve(pairs, thresholds=thresholds)


def _coverage_at(thresholds: list[ThresholdRecommendation], target: float) -> float | None:
    for t in thresholds:
        if abs(t.target_risk - target) < 1e-12:
            return t.coverage
    return None


def _gate_verdict(
    raw: ConditionResult,
    calibrated: ConditionResult | None,
    cert_level: CertifiedLevel | None,
    *,
    verdict_alpha: float,
    min_coverage: float,
    cert_error: str | None = None,
) -> GateVerdict:
    """
    Verdict from the CRC certificate at `verdict_alpha`; ECE is context only.

    Calibration and automation are independent axes (findings §3): a model can
    reach excellent ECE and still certify nothing, or be well calibrated out of
    the box and automate most traffic. So ECE never moves the verdict.
    """
    notes: list[str] = []

    cert = cert_level.certificate if cert_level is not None else None
    feasible = bool(cert is not None and cert.feasible and cert_level.test is not None)
    coverage = float(cert_level.test["coverage"]) if feasible else None
    joint = float(cert_level.test["joint_error_rate"]) if feasible else None

    if not feasible:
        recommendation = "no-go"
        if cert_error is not None:
            notes.append(f"No certificate at {verdict_alpha:.0%}: {cert_error}")
        else:
            notes.append(
                f"No certified threshold at {verdict_alpha:.0%} joint automated error "
                "rate: nothing can run automatically at this risk."
            )
            if cert is not None:
                notes.extend(cert.notes[:1])
    elif coverage >= min_coverage:
        recommendation = "go"
        notes.append(
            f"Certified at {verdict_alpha:.0%}: {coverage:.1%} of test traffic auto-accepted "
            f"(>= {min_coverage:.0%})."
        )
    else:
        recommendation = "watch"
        notes.append(
            f"Certified at {verdict_alpha:.0%}, but only {coverage:.1%} of test traffic is "
            f"auto-accepted (< {min_coverage:.0%}). Savings may not justify integration."
        )

    ece_improved = False
    rel_drop: float | None = None
    if calibrated is not None and raw.ece > 1e-12:
        rel_drop = (raw.ece - calibrated.ece) / raw.ece
        ece_improved = calibrated.ece < raw.ece - 1e-6
        notes.append(
            f"Context: temperature scaling changed ECE {raw.ece:.3f} -> {calibrated.ece:.3f} "
            f"({rel_drop:+.0%} drop). Not used in the verdict."
        )

    focus = calibrated or raw
    return GateVerdict(
        recommendation=recommendation,
        verdict_alpha=verdict_alpha,
        min_coverage=min_coverage,
        certified_feasible=feasible,
        certified_threshold=cert.threshold if feasible else None,
        certified_coverage=coverage,
        certified_joint_error=joint,
        ece_improved=ece_improved,
        ece_relative_drop=rel_drop,
        coverage_at_1pct=_coverage_at(focus.thresholds, 0.01),
        coverage_at_5pct=_coverage_at(focus.thresholds, 0.05),
        notes=notes,
    )


def run_gate(
    cases: list[DecisionCase],
    predictions: list[DecisionPrediction],
    *,
    cal_ratio: float = 0.3,
    seed: int = 42,
    n_bins: int = 10,
    target_risks: list[float] | None = None,
    calibrate: bool = True,
    provider: str | None = None,
    model: str | None = None,
    contract: DecisionContract | None = None,
    verdict_alpha: float = 0.05,
    min_coverage: float = 0.30,
    temp_ratio: float = 0.2,
) -> GateReport:
    """
    Technical gate protocol:

    1. Split pairs into calibration / test
    2. Fit temperature (optional) on calibration
    3. Choose risk thresholds on calibration
    4. Report metrics + threshold performance on test only
    5. Verdict from a CRC certificate at `verdict_alpha`, computed with the same
       seed / cal_ratio / temp_ratio / contract as `decisionops certify`, so the
       two commands agree on feasibility and coverage.

    With a `contract`, thresholds and risk-coverage count only errors on the
    automatic choices; escalations are never risk. Calibration metrics (ECE,
    Brier, NLL) are unaffected — they describe the probabilities, not the policy.
    """
    if not 0.0 < verdict_alpha < 1.0:
        raise ValueError(f"verdict_alpha must be in (0, 1), got {verdict_alpha}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    if target_risks is None:
        target_risks = [0.01, 0.05, 0.10]
    if contract is not None:
        contract.validate_data(cases, predictions)
    auto = auto_set_of(contract)

    pairs = align_cases_and_predictions(cases, predictions)
    split = split_pairs(pairs, cal_ratio=cal_ratio, seed=seed)

    preds_all = [p for _, p in pairs]
    providers = {p.provider for p in preds_all if p.provider}
    models = {p.model for p in preds_all if p.model}
    inferred_provider = provider or (next(iter(providers)) if len(providers) == 1 else None)
    inferred_model = model or (next(iter(models)) if len(models) == 1 else None)

    conditions: list[ConditionResult] = []
    temperature_value: float | None = None
    temperature_fit: dict = {}
    isotonic_fit: dict = {}

    def _condition_on_split(
        cal_pairs: list[tuple[DecisionCase, DecisionPrediction]],
        test_pairs: list[tuple[DecisionCase, DecisionPrediction]],
        *,
        name: str,
        temperature: float | None = None,
    ) -> ConditionResult:
        cal_recs = recommend_thresholds(cal_pairs, target_risks=target_risks, auto_choices=auto)
        fixed = [(r.target_risk, r.threshold) for r in cal_recs]
        test_recs = evaluate_fixed_thresholds(
            test_pairs, target_risks_and_thresholds=fixed, auto_choices=auto
        )
        return _pairs_metrics(
            test_pairs,
            n_bins=n_bins,
            thresholds=test_recs,
            cal_thresholds=cal_recs,
            name=name,
            temperature=temperature,
            auto_choices=auto,
        )

    # Raw condition
    raw_condition = _condition_on_split(split.calibration, split.test, name="raw")
    conditions.append(raw_condition)

    calibrated_condition: ConditionResult | None = None
    if calibrate:
        ts = fit_temperature(split.calibration)
        temperature_value = ts.temperature
        temperature_fit = {
            "temperature": ts.temperature,
            "n_calibration": ts.n_calibration,
            "nll_before": ts.nll_before,
            "nll_after": ts.nll_after,
        }
        cal_scaled = apply_temperature_to_pairs(split.calibration, ts.temperature)
        test_scaled = apply_temperature_to_pairs(split.test, ts.temperature)
        calibrated_condition = _condition_on_split(
            cal_scaled,
            test_scaled,
            name="temperature_scaled",
            temperature=ts.temperature,
        )
        conditions.append(calibrated_condition)

        iso = fit_isotonic(split.calibration)
        isotonic_fit = {
            "n_calibration": iso.n_calibration,
            "nll_before": iso.nll_before,
            "nll_after": iso.nll_after,
        }
        cal_iso = apply_isotonic_to_pairs(split.calibration, iso)
        test_iso = apply_isotonic_to_pairs(split.test, iso)
        conditions.append(
            _condition_on_split(cal_iso, test_iso, name="isotonic")
        )

    cert_level: CertifiedLevel | None = None
    cert_error: str | None = None
    try:
        cert = run_certification(
            cases,
            predictions,
            alphas=[verdict_alpha],
            cal_ratio=cal_ratio,
            temp_ratio=temp_ratio if calibrate else 0.0,
            seed=seed,
            calibrate=calibrate,
            contract=contract,
        )
        cert_level = cert.levels[0]
    except ValueError as exc:
        # Pools too small for the three-way split: no certificate, not a crash.
        cert_error = str(exc)

    verdict = _gate_verdict(
        raw_condition,
        calibrated_condition,
        cert_level,
        verdict_alpha=verdict_alpha,
        min_coverage=min_coverage,
        cert_error=cert_error,
    )

    return GateReport(
        split=SplitInfo(
            seed=split.seed,
            cal_ratio=split.cal_ratio,
            n_total=split.n_total,
            n_calibration=split.n_calibration,
            n_test=split.n_test,
        ),
        provider=inferred_provider,
        model=inferred_model,
        target_risks=list(target_risks),
        n_bins=n_bins,
        temperature=temperature_value,
        temperature_fit=temperature_fit,
        isotonic_fit=isotonic_fit,
        conditions=conditions,
        verdict=verdict,
        latency=latency_stats(preds_all),
        metadata={
            "protocol": "cal_test_split",
            "verdict_basis": "crc_certificate",
            "contract": contract.summary() if contract is not None else None,
        },
    )
