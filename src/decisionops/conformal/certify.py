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
from decisionops.contract import DecisionContract, auto_set_of
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
    loss_bound: float | None = None,
    contract: DecisionContract | None = None,
    confidence_level: float | None = None,
) -> CertificationReport:
    """
    Three-way split: fit temperature, derive CRC thresholds, verify — on disjoint data.

    `confidence_level` (e.g. 0.95) switches every level from an in-expectation
    CRC certificate to a high-probability one; see `certify_threshold`.

    Temperature must NOT be fit on the CRC calibration set. Doing so makes those
    probabilities optimistically biased, breaks exchangeability with test, and the
    finite-sample guarantee silently fails. `temp_ratio` carves the temperature-fit
    portion out of the non-test data; set `calibrate=False` to skip it entirely.

    With a `contract`, only errors on its automatic choices are risk: the
    certified quantity becomes the automated miss rate (auto-executed AND wrong,
    over all traffic). Escalations are free of risk by construction. Contract
    `severity` weights each miss by its true label and sets the CRC loss bound
    B unless `loss_bound` is passed explicitly.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10]
    if not 0.0 <= temp_ratio < 1.0:
        raise ValueError(f"temp_ratio must be in [0, 1), got {temp_ratio}")
    if contract is not None:
        contract.validate_data(cases, predictions)
    auto = auto_set_of(contract)
    severity_of = contract.miss_severity if contract is not None else None
    if loss_bound is None:
        loss_bound = contract.loss_bound() if contract is not None else 1.0
    elif loss_bound <= 0.0:
        raise ValueError(f"loss_bound must be > 0, got {loss_bound}")

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
        cert = certify_threshold(
            cal_pairs,
            alpha,
            loss_bound=loss_bound,
            auto_choices=auto,
            severity_of=severity_of,
            confidence_level=confidence_level,
        )
        test_stats = None
        held = None
        if cert.feasible and cert.threshold is not None:
            test_stats = evaluate_certified_threshold(
                test_pairs,
                cert.threshold,
                auto_choices=auto,
                severity_of=severity_of,
            )
            held = bool(test_stats["joint_error_rate"] <= alpha)
        levels.append(
            CertifiedLevel(certificate=cert, test=test_stats, bound_held_on_test=held)
        )

    if contract is not None:
        if contract.severity:
            notes.append(
                f"Contract {contract.name!r}: only auto choices "
                f"{sorted(contract.auto_set)} count as risk; miss severity "
                f"{dict(sorted(contract.severity.items()))} (B={loss_bound:g}). "
                "Certified quantity is the severity-weighted automated miss rate."
            )
        else:
            notes.append(
                f"Contract {contract.name!r}: only auto choices "
                f"{sorted(contract.auto_set)} count as risk; escalations never do. "
                "Certified quantity is the automated miss rate."
            )
    notes.append(
        "Certified quantity is the (possibly severity-weighted) joint automated "
        "error, not conditional selective risk. Assumes exchangeability."
    )
    if confidence_level is None:
        notes += [
            "CRC guarantee is on E[loss], marginal over calibration and test draws.",
            "A single test split exceeding alpha is expected roughly half the time: "
            "the smallest feasible lambda puts true risk just under alpha. Validate "
            "by averaging over seeds.",
        ]
    else:
        notes.append(
            f"High-probability guarantee: with probability >= {confidence_level:.0%} over "
            "the calibration draw, true joint error <= alpha. A test split can still "
            "land above alpha by sampling noise."
        )
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
            "guarantee": "expectation" if confidence_level is None else "high_probability",
            "confidence_level": confidence_level,
            "loss_bound": loss_bound,
            "temp_ratio": temp_ratio,
            "protocol": "three_way_split_temp_crc_test",
            "contract": contract.summary() if contract is not None else None,
            "contract_hash": contract.fingerprint() if contract is not None else None,
        },
    )
