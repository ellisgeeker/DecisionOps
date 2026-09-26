"""Split-aware certification run: calibrate, certify on cal, verify on test."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from decisionops.calibration.temperature import (
    apply_temperature_to_pairs,
    fit_temperature,
)
from decisionops.conformal.risk_control import (
    CRCResult,
    certify_threshold,
    evaluate_certified_threshold,
)
from decisionops.io import align_cases_and_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction, SplitInfo
from decisionops.split import split_pairs


class CertifiedLevel(BaseModel):
    """One target risk level: certificate from cal, measurement on test."""

    certificate: CRCResult
    test: dict[str, float | int] | None = None
    bound_held_on_test: bool | None = None
    """
    Whether this one test split came out <= alpha.

    Expect roughly a coin flip even when the implementation is correct: CRC picks
    the smallest feasible lambda, so true risk sits just under alpha and a single
    empirical draw lands on either side. Judge correctness by the mean across
    seeds, never by one split.
    """


class CertificationReport(BaseModel):
    split: SplitInfo
    provider: str | None = None
    model: str | None = None
    calibrated: bool = False
    temperature: float | None = None
    n_temperature_fit: int = 0
    """Samples used to fit temperature. Disjoint from the CRC calibration set."""

    levels: list[CertifiedLevel]
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def run_certification(
    cases: list[DecisionCase],
    predictions: list[DecisionPrediction],
    *,
    alphas: list[float] | None = None,
    cal_ratio: float = 0.3,
    temp_ratio: float = 0.2,
    seed: int = 42,
    calibrate: bool = True,
    loss_bound: float = 1.0,
) -> CertificationReport:
    """
    Three-way split: fit temperature, derive CRC thresholds, verify — on disjoint data.

    Temperature must NOT be fit on the CRC calibration set. Doing so makes those
    probabilities optimistically biased, breaks exchangeability with test, and the
    finite-sample guarantee silently fails. `temp_ratio` carves the temperature-fit
    portion out of the non-test data; set `calibrate=False` to skip it entirely.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10]
    if not 0.0 <= temp_ratio < 1.0:
        raise ValueError(f"temp_ratio must be in [0, 1), got {temp_ratio}")

    pairs = align_cases_and_predictions(cases, predictions)
    split = split_pairs(pairs, cal_ratio=cal_ratio, seed=seed)

    preds_all = [p for _, p in pairs]
    providers = {p.provider for p in preds_all if p.provider}
    models = {p.model for p in preds_all if p.model}

    cal_pairs = split.calibration
    test_pairs = split.test
    temperature: float | None = None
    n_temp_fit = 0
    notes: list[str] = []

    if calibrate and temp_ratio > 0.0:
        # Carve the temperature-fit slice out of the calibration pool so the CRC
        # set stays untouched by the fit.
        inner = split_pairs(cal_pairs, cal_ratio=temp_ratio, seed=seed + 1)
        temp_pairs, cal_pairs = inner.calibration, inner.test
        n_temp_fit = len(temp_pairs)
        ts = fit_temperature(temp_pairs)
        temperature = ts.temperature
        cal_pairs = apply_temperature_to_pairs(cal_pairs, ts.temperature)
        test_pairs = apply_temperature_to_pairs(test_pairs, ts.temperature)
        notes.append(
            f"Temperature T={temperature:.4f} fit on {n_temp_fit} held-out samples, "
            f"disjoint from the {len(cal_pairs)} used for CRC."
        )
    elif calibrate:
        notes.append(
            "temp_ratio=0: temperature skipped. Fitting it on the CRC set would "
            "void the guarantee."
        )

    levels: list[CertifiedLevel] = []
    for alpha in alphas:
        cert = certify_threshold(cal_pairs, alpha, loss_bound=loss_bound)
        test_stats = None
        held = None
        if cert.feasible and cert.threshold is not None:
            test_stats = evaluate_certified_threshold(test_pairs, cert.threshold)
            held = bool(test_stats["joint_error_rate"] <= alpha)
        levels.append(
            CertifiedLevel(certificate=cert, test=test_stats, bound_held_on_test=held)
        )

    notes += [
        "CRC certifies the joint automated error rate, not conditional selective risk.",
        "Guarantee is marginal over calibration and test draws, and assumes exchangeability.",
        "A single test split exceeding alpha is expected roughly half the time: "
        "the guarantee is on E[loss], and the smallest feasible lambda puts true "
        "risk just under alpha. Validate by averaging over seeds.",
    ]
    infeasible = [
        f"{level.certificate.alpha:.0%}" for level in levels if not level.certificate.feasible
    ]
    if infeasible:
        notes.append(
            f"No certified threshold exists at: {', '.join(infeasible)} "
            f"(n_crc_cal={len(cal_pairs)})."
        )

    return CertificationReport(
        split=SplitInfo(
            seed=split.seed,
            cal_ratio=split.cal_ratio,
            n_total=split.n_total,
            n_calibration=len(cal_pairs),
            n_test=split.n_test,
        ),
        provider=next(iter(providers)) if len(providers) == 1 else None,
        model=next(iter(models)) if len(models) == 1 else None,
        calibrated=calibrate and temp_ratio > 0.0,
        temperature=temperature,
        n_temperature_fit=n_temp_fit,
        levels=levels,
        notes=notes,
        metadata={
            "alphas": list(alphas),
            "loss_bound": loss_bound,
            "temp_ratio": temp_ratio,
            "protocol": "three_way_split_temp_crc_test",
        },
    )
