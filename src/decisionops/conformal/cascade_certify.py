"""Split-aware two-stage cascade certification: temperature, LTT, test check."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from decisionops.calibration.temperature import (
    apply_temperature_to_pairs,
    fit_temperature,
)
from decisionops.cascade import escalate_cost
from decisionops.conformal.cascade_risk import (
    CASCADE_QUANTITY,
    CascadeCertificate,
    CascadePoint,
    certify_cascade,
    measure_cascade,
    quantile_grid,
    stage_arrays_from_pairs,
)
from decisionops.contract import DecisionContract, auto_set_of
from decisionops.io import align_cases_and_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction, SplitInfo
from decisionops.split import split_pairs

DEFAULT_GRID_SIZE = 12
DEFAULT_CONFIDENCE_LEVEL = 0.9


class CascadeStageFit(BaseModel):
    """Identity and temperature for one cascade stage after the split."""

    name: str
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    n_temperature_fit: int = 0


class CascadeCertificationReport(BaseModel):
    """One alpha: cascade certificate from cal, measurement on test."""

    split: SplitInfo
    stages: list[CascadeStageFit]
    certificate: CascadeCertificate
    test: CascadePoint | None = None
    bound_held_on_test: bool | None = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _reorder(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    order: list[tuple[DecisionCase, DecisionPrediction]],
) -> list[tuple[DecisionCase, DecisionPrediction]]:
    by_id = {case.id: (case, pred) for case, pred in pairs}
    missing = [case.id for case, _ in order if case.id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"stage is missing predictions for ids: {preview}")
    return [by_id[case.id] for case, _ in order]


def _identity(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> tuple[str | None, str | None]:
    preds = [p for _, p in pairs]
    providers = {p.provider for p in preds if p.provider}
    models = {p.model for p in preds if p.model}
    provider = next(iter(providers)) if len(providers) == 1 else None
    model = next(iter(models)) if len(models) == 1 else None
    return provider, model


def _eligible_confidences(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    auto: frozenset[str] | None,
) -> list[float]:
    out: list[float] = []
    for _, pred in pairs:
        if auto is None or pred.prediction in auto:
            out.append(pred.confidence)
    return out


def run_cascade_certification(
    cases: list[DecisionCase],
    stage1: list[DecisionPrediction],
    stage2: list[DecisionPrediction],
    contract: DecisionContract,
    *,
    stage_names: tuple[str, str] = ("stage1", "stage2"),
    alpha: float = 0.05,
    cal_ratio: float = 0.3,
    temp_ratio: float = 0.2,
    seed: int = 42,
    calibrate: bool = True,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    grid_size: int = DEFAULT_GRID_SIZE,
    stage_costs: tuple[float, float] | None = None,
) -> CascadeCertificationReport:
    """
    Three-way split, then Learn-Then-Test on a 2-D threshold grid.

    Temperature for each stage is fit on a slice carved out of calibration
    (disjoint from the LTT set). Grids are quantiles of that slice's
    confidences — labels are not used — so the hypotheses stay pre-specified.
    `--no-calibrate` / `temp_ratio=0` skips the fit and falls back to an even
    grid on (0, 1).
    """
    if contract.severity:
        raise ValueError(
            "cascade certification needs 0-1 loss; contract severity is not supported"
        )
    if contract.is_ordinal:
        raise ValueError(
            "cascade certification needs equality 0-1 loss; ordinal under-rate "
            "loss is not supported"
        )
    if not 0.0 <= temp_ratio < 1.0:
        raise ValueError(f"temp_ratio must be in [0, 1), got {temp_ratio}")
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")

    contract.validate_data(cases, stage1)
    contract.validate_data(cases, stage2)
    auto = auto_set_of(contract)
    pairs1 = align_cases_and_predictions(cases, stage1)
    pairs2 = align_cases_and_predictions(cases, stage2)

    split = split_pairs(pairs1, cal_ratio=cal_ratio, seed=seed)
    cal1, test1 = split.calibration, split.test
    cal2, test2 = _reorder(pairs2, cal1), _reorder(pairs2, test1)

    notes: list[str] = []
    t1 = t2 = None
    n_temp = 0
    grid_source = "unit_interval"
    grid_pairs1, grid_pairs2 = cal1, cal2

    if calibrate and temp_ratio > 0.0:
        inner = split_pairs(cal1, cal_ratio=temp_ratio, seed=seed + 1)
        temp1, cal1 = inner.calibration, inner.test
        temp2, cal2 = _reorder(cal2, temp1), _reorder(cal2, cal1)
        n_temp = len(temp1)
        t1 = fit_temperature(temp1).temperature
        t2 = fit_temperature(temp2).temperature
        cal1 = apply_temperature_to_pairs(cal1, t1)
        cal2 = apply_temperature_to_pairs(cal2, t2)
        test1 = apply_temperature_to_pairs(test1, t1)
        test2 = apply_temperature_to_pairs(test2, t2)
        grid_pairs1 = apply_temperature_to_pairs(temp1, t1)
        grid_pairs2 = apply_temperature_to_pairs(temp2, t2)
        grid_source = "temperature_slice"
        notes.append(
            f"Temperature T1={t1:.4f}, T2={t2:.4f} fit on {n_temp} held-out samples, "
            f"disjoint from the {len(cal1)} used for cascade LTT."
        )
    elif calibrate:
        notes.append(
            "temp_ratio=0: temperature skipped. Fitting it on the LTT set would "
            "void the guarantee. Threshold grids fall back to (0, 1)."
        )

    if grid_source == "temperature_slice":
        g1 = quantile_grid(_eligible_confidences(grid_pairs1, auto), grid_size)
        g2 = quantile_grid(_eligible_confidences(grid_pairs2, auto), grid_size)
    else:
        g1 = quantile_grid([], grid_size)
        g2 = quantile_grid([], grid_size)
        notes.append(
            "Threshold grids are evenly spaced on (0, 1); pass --calibrate with "
            "temp_ratio>0 to take quantiles from a label-free temperature slice."
        )

    priced = bool(contract.costs) and stage_costs is not None
    if stage_costs is not None and not contract.costs:
        raise ValueError("--stage-cost needs contract.costs for every choice")
    if priced:
        missing = [c for c in contract.choices if c not in contract.costs]
        if missing:
            raise ValueError(
                f"contract {contract.name!r} is missing costs for {missing}; "
                "cascade cost objective needs a cost for every choice"
            )
        if not contract.escalate_choices:
            raise ValueError(
                f"contract {contract.name!r} has no escalate choices; "
                "cascade needs at least one non-auto path to price against"
            )

    path_costs = dict(contract.costs) if priced else None
    s1 = stage_arrays_from_pairs(cal1, auto_choices=auto, path_costs=path_costs)
    s2 = stage_arrays_from_pairs(cal2, auto_choices=auto, path_costs=path_costs)
    esc = None
    if priced:
        esc = np.array([escalate_cost(contract, case) for case, _ in cal1], dtype=np.float64)

    cert = certify_cascade(
        s1,
        s2,
        alpha,
        lambda1_grid=g1,
        lambda2_grid=g2,
        confidence_level=confidence_level,
        stage_costs=stage_costs if priced else None,
        escalate_costs=esc,
    )

    test_point = None
    held = None
    if cert.feasible:
        ts1 = stage_arrays_from_pairs(test1, auto_choices=auto, path_costs=path_costs)
        ts2 = stage_arrays_from_pairs(test2, auto_choices=auto, path_costs=path_costs)
        tes = None
        if priced:
            tes = np.array(
                [escalate_cost(contract, case) for case, _ in test1], dtype=np.float64
            )
        test_point = measure_cascade(
            ts1,
            ts2,
            cert.lambda1,
            cert.lambda2,
            stage_costs=stage_costs if priced else None,
            escalate_costs=tes,
        )
        held = bool(test_point.joint_error_rate <= alpha)

    notes.extend(cert.notes)
    notes.append(
        f"Certified quantity is the cascade joint automated error ({CASCADE_QUANTITY}). "
        "High-probability only (Learn-Then-Test, Bonferroni rows, fixed-sequence lambda2)."
    )

    p1, m1 = _identity(pairs1)
    p2, m2 = _identity(pairs2)
    stages = [
        CascadeStageFit(
            name=stage_names[0], provider=p1, model=m1, temperature=t1, n_temperature_fit=n_temp
        ),
        CascadeStageFit(
            name=stage_names[1], provider=p2, model=m2, temperature=t2, n_temperature_fit=n_temp
        ),
    ]
    return CascadeCertificationReport(
        split=SplitInfo(
            seed=split.seed,
            cal_ratio=split.cal_ratio,
            n_total=split.n_total,
            n_calibration=len(cal1),
            n_test=split.n_test,
        ),
        stages=stages,
        certificate=cert,
        test=test_point,
        bound_held_on_test=held,
        notes=notes,
        metadata={
            "alpha": alpha,
            "guarantee": "high_probability",
            "confidence_level": confidence_level,
            "risk": "cascade_joint",
            "grid_size": grid_size,
            "grid_source": grid_source,
            "temp_ratio": temp_ratio,
            "protocol": "three_way_split_temp_cascade_ltt_test",
            "contract": contract.summary(),
            "contract_hash": contract.fingerprint(),
            "stage_costs": list(stage_costs) if stage_costs is not None else None,
        },
    )
