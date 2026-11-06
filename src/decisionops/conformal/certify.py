"""Split-aware certification run: calibrate, certify on cal, verify on test."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from decisionops.calibration.temperature import (
    apply_temperature_to_pairs,
    fit_temperature,
)
from decisionops.conformal.risk_control import (
    ORDINAL_QUANTITY,
    CRCResult,
    certify_selective_threshold,
    certify_threshold,
    evaluate_certified_threshold,
)
from decisionops.contract import DecisionContract, auto_set_of
from decisionops.io import align_cases_and_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction, SplitInfo
from decisionops.split import split_pairs

RiskKind = Literal["joint", "selective"]
SELECTIVE_DEFAULT_LEVEL = 0.9


class LevelStability(BaseModel):
    """
    Spread of one alpha level over re-splits of the same pool.

    Each split reruns the whole procedure (temperature, CRC, test) on a fresh
    seed. It measures split-to-split variation inside *this* pool, not a
    population expectation, but it shows how much a single certificate can move.
    """

    n_splits: int
    n_feasible: int
    threshold: dict[str, float] | None = None
    """min / p10 / median / p90 / max of the certified threshold."""

    test_coverage: dict[str, float] | None = None
    test_joint_error: dict[str, float] | None = None
    mean_test_joint_error: float | None = None
    """Mean over feasible splits: what an in-expectation certificate promises <= alpha."""


def _spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    import numpy as np

    q = np.quantile(values, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {k: float(v) for k, v in zip(("min", "p10", "median", "p90", "max"), q, strict=True)}


class CertifiedLevel(BaseModel):
    """One target risk level: certificate from cal, measurement on test."""

    certificate: CRCResult
    test: dict[str, float | int] | None = None
    bound_held_on_test: bool | None = None
    stability: LevelStability | None = None
    """Re-split spread; set when `run_certification(stability_splits=N)`."""
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
    risk: RiskKind = "joint",
    stability_splits: int = 0,
) -> CertificationReport:
    """
    Three-way split: fit temperature, derive CRC thresholds, verify — on disjoint data.

    `risk="joint"` (default) bounds the share of *all* traffic that is
    auto-accepted and wrong. `risk="selective"` bounds the error rate *among*
    auto-accepted cases (Learn-Then-Test); it is high-probability only, so
    `confidence_level` defaults to 0.9, and it needs 0-1 loss.

    `stability_splits=N` reruns the whole procedure on N further seeds and
    attaches the spread of threshold, test coverage and test joint error to
    each level. Use it when the calibration set holds few errors: a valid
    certificate can still move a lot from one split to the next.

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
    if stability_splits < 0:
        raise ValueError(f"stability_splits must be >= 0, got {stability_splits}")
    if not 0.0 <= temp_ratio < 1.0:
        raise ValueError(f"temp_ratio must be in [0, 1), got {temp_ratio}")
    if risk not in ("joint", "selective"):
        raise ValueError(f"risk must be 'joint' or 'selective', got {risk!r}")
    if risk == "selective":
        if contract is not None and contract.severity:
            raise ValueError(
                "selective risk is an error *rate* among accepted cases and needs 0-1 "
                "loss; contract severity is not supported with risk='selective'"
            )
        if contract is not None and contract.is_ordinal:
            raise ValueError(
                "selective risk needs equality 0-1 loss among accepted cases; "
                "ordinal under-rate loss is not supported with risk='selective'"
            )
        if loss_bound not in (None, 1.0):
            raise ValueError(f"selective risk needs 0-1 loss, got loss_bound={loss_bound}")
        if confidence_level is None:
            confidence_level = SELECTIVE_DEFAULT_LEVEL
    if contract is not None:
        contract.validate_data(cases, predictions)
    auto = auto_set_of(contract)
    severity_of = None
    loss_of = None
    quantity = None
    if contract is not None and contract.is_ordinal:
        loss_of = contract.pair_loss
        quantity = ORDINAL_QUANTITY
    elif contract is not None and contract.severity:
        severity_of = contract.miss_severity
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
        if risk == "selective":
            cert = certify_selective_threshold(
                cal_pairs, alpha, confidence_level=confidence_level, auto_choices=auto
            )
        else:
            cert = certify_threshold(
                cal_pairs,
                alpha,
                loss_bound=loss_bound,
                auto_choices=auto,
                severity_of=severity_of,
                loss_of=loss_of,
                controlled_quantity=quantity,
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
                loss_of=loss_of,
            )
            measured = "selective_risk" if risk == "selective" else "joint_error_rate"
            value = float(test_stats[measured])
            # No accepted test case: selective risk is undefined, nothing to compare.
            held = None if value != value else bool(value <= alpha)
        levels.append(
            CertifiedLevel(certificate=cert, test=test_stats, bound_held_on_test=held)
        )

    if contract is not None:
        if contract.is_ordinal:
            notes.append(
                f"Contract {contract.name!r}: ordinal under-rate loss on order "
                f"{contract.choice_order} with under_rate_tolerance="
                f"{contract.under_rate_tolerance}; only auto choices "
                f"{sorted(contract.auto_set)} count as risk. Over-rating is free."
            )
        elif contract.severity:
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
    if risk == "selective":
        notes.append(
            "Certified quantity is the selective error rate: errors among auto-accepted "
            "cases only (Learn-Then-Test, fixed-sequence Clopper-Pearson). Assumes "
            "exchangeability."
        )
    else:
        if contract is not None and contract.is_ordinal:
            kind = "ordinal under-rate joint automated error"
        elif contract is not None and contract.severity:
            kind = "severity-weighted joint automated error"
        else:
            kind = "joint automated error rate"
        notes.append(
            f"Certified quantity is the {kind}, not conditional selective risk. "
            "Assumes exchangeability."
        )
    if risk == "selective":
        notes.append(
            f"Selective guarantee: with probability >= {confidence_level:.0%} over the "
            "calibration draw, the error rate among auto-accepted cases is <= alpha. "
            "A test split can still land above alpha by sampling noise."
        )
    elif confidence_level is None:
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
    from decisionops.conformal.risk_control import FEW_ERRORS

    thin = [
        level.certificate
        for level in levels
        if level.certificate.feasible
        and level.certificate.error_budget is not None
        and level.certificate.error_budget < FEW_ERRORS
    ]
    if thin:
        notes.append(
            "Few errors carry the certificate: the bound tolerates only "
            + ", ".join(f"{c.error_budget:.1f} at alpha={c.alpha:g}" for c in thin)
            + f" on n_cal={thin[0].n_calibration}. Valid, but one case can move the "
            "threshold; run with --stability-splits to see how far."
        )
    infeasible = [
        f"{level.certificate.alpha:.0%}" for level in levels if not level.certificate.feasible
    ]
    if infeasible:
        notes.append(
            f"No certified threshold exists at: {', '.join(infeasible)} "
            f"(n_crc_cal={len(cal_pairs)})."
        )

    if stability_splits:
        reruns = [
            run_certification(
                cases,
                predictions,
                alphas=alphas,
                cal_ratio=cal_ratio,
                temp_ratio=temp_ratio,
                seed=seed + 1000 + i,
                calibrate=calibrate,
                loss_bound=loss_bound,
                contract=contract,
                confidence_level=confidence_level,
                risk=risk,
            )
            for i in range(stability_splits)
        ]
        measured = "selective_risk" if risk == "selective" else "joint_error_rate"
        for j, level in enumerate(levels):
            runs = [r.levels[j] for r in reruns]
            ok = [r for r in runs if r.certificate.feasible and r.test is not None]
            errs = [float(r.test[measured]) for r in ok if r.test[measured] == r.test[measured]]
            level.stability = LevelStability(
                n_splits=stability_splits,
                n_feasible=len(ok),
                threshold=_spread([float(r.certificate.threshold) for r in ok]),
                test_coverage=_spread([float(r.test["coverage"]) for r in ok]),
                test_joint_error=_spread(errs),
                mean_test_joint_error=(sum(errs) / len(errs)) if errs else None,
            )
            cov = level.stability.test_coverage
            if cov and cov["p90"] - cov["p10"] > 0.2:
                notes.append(
                    f"alpha={level.certificate.alpha:g}: certified test coverage moves "
                    f"from {cov['p10']:.0%} to {cov['p90']:.0%} (p10-p90) over "
                    f"{stability_splits} re-splits. One certificate is a noisy draw here; "
                    "label more data, or set the operating point from the spread."
                )
            if len(ok) < stability_splits:
                notes.append(
                    f"alpha={level.certificate.alpha:g}: certified on {len(ok)} of "
                    f"{stability_splits} re-splits."
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
            "risk": risk,
            "temp_ratio": temp_ratio,
            "stability_splits": stability_splits,
            "protocol": "three_way_split_temp_crc_test",
            "contract": contract.summary() if contract is not None else None,
            "contract_hash": contract.fingerprint() if contract is not None else None,
        },
    )
