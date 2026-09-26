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
) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for p in risk_coverage_curve(pairs):
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
        risk_coverage=_curve_payload(pairs),
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


def _gate_verdict(raw: ConditionResult, calibrated: ConditionResult | None) -> GateVerdict:
    notes: list[str] = []
    ece_improved = False
    rel_drop: float | None = None

    if calibrated is not None and raw.ece > 1e-12:
        rel_drop = (raw.ece - calibrated.ece) / raw.ece
        ece_improved = calibrated.ece < raw.ece - 1e-6
        if ece_improved and rel_drop is not None and rel_drop >= 0.15:
            notes.append(f"ECE improved by {rel_drop:.0%} after temperature scaling.")
        elif ece_improved:
            notes.append("ECE improved only marginally after temperature scaling.")
        else:
            notes.append("Temperature scaling did not improve ECE on the test split.")
    elif calibrated is None:
        notes.append("Calibration was not run; verdict based on raw probabilities only.")

    focus = calibrated or raw
    cov1 = _coverage_at(focus.thresholds, 0.01)
    cov5 = _coverage_at(focus.thresholds, 0.05)

    in_band: bool | None = None
    if cov1 is not None:
        # Band from the feasibility analysis: automation neither empty nor trivial.
        in_band = 0.15 <= cov1 <= 0.75
        if cov1 < 0.05:
            notes.append(
                f"@1% risk coverage is very low ({cov1:.1%}); little room to automate."
            )
        elif cov1 > 0.95:
            notes.append(
                f"@1% risk coverage is very high ({cov1:.1%}); "
                "thresholding may be unnecessary."
            )
        elif in_band:
            notes.append(f"@1% risk coverage={cov1:.1%} sits in a usable automation band.")
        else:
            notes.append(f"@1% risk coverage={cov1:.1%} is outside the preferred 15–75% band.")

    # Recommendation heuristic (not a business decision).
    strong = (
        calibrated is not None
        and ece_improved
        and rel_drop is not None
        and rel_drop >= 0.15
        and in_band
    )
    mild = (calibrated is not None and (ece_improved or in_band)) or (
        calibrated is None and in_band
    )
    if strong:
        recommendation = "go"
    elif mild:
        recommendation = "watch"
    else:
        recommendation = "no-go"

    return GateVerdict(
        ece_improved=ece_improved,
        ece_relative_drop=rel_drop,
        coverage_at_1pct=cov1,
        coverage_at_5pct=cov5,
        coverage_in_band_1pct=in_band,
        notes=notes,
        recommendation=recommendation,
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
) -> GateReport:
    """
    Technical gate protocol:

    1. Split pairs into calibration / test
    2. Fit temperature (optional) on calibration
    3. Choose risk thresholds on calibration
    4. Report metrics + threshold performance on test only
    """
    if target_risks is None:
        target_risks = [0.01, 0.05, 0.10]

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
        cal_recs = recommend_thresholds(cal_pairs, target_risks=target_risks)
        fixed = [(r.target_risk, r.threshold) for r in cal_recs]
        test_recs = evaluate_fixed_thresholds(test_pairs, target_risks_and_thresholds=fixed)
        return _pairs_metrics(
            test_pairs,
            n_bins=n_bins,
            thresholds=test_recs,
            cal_thresholds=cal_recs,
            name=name,
            temperature=temperature,
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

    verdict = _gate_verdict(raw_condition, calibrated_condition)

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
        metadata={"protocol": "cal_test_split"},
    )
