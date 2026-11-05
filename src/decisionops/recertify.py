"""
Recertify a live profile from random audit samples only.

After go-live, organic labels come only from escalations (selected by low
confidence) and cannot recalibrate CRC unweighted. `decide()` draws an audit
sample independent of the score (`Decision.audit=True`); those rows get the
expensive verdict and are exchangeable with live traffic.

`run_recertification` joins decisions × outcomes × predictions, keeps only
audit rows whose labels have matured, applies the profile's frozen temperature
(does not re-fit T — that would consume audit mass and break exchangeability
with the original certificate's calibrator), and runs CRC on that set alone.
The result is compared to the current profile so an operator can keep, replace,
or escalate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from decisionops.calibration.temperature import apply_temperature
from decisionops.cascade import escalate_cost
from decisionops.conformal.cascade_certify import DEFAULT_GRID_SIZE
from decisionops.conformal.cascade_risk import (
    CascadeCertificate,
    certify_cascade,
    quantile_grid,
    stage_arrays_from_pairs,
)
from decisionops.conformal.risk_control import (
    ORDINAL_QUANTITY,
    SELECTIVE_QUANTITY,
    CRCResult,
    certify_selective_threshold,
    certify_threshold,
    evaluate_certified_threshold,
)
from decisionops.contract import DecisionContract, auto_set_of
from decisionops.decide import (
    CascadeProfile,
    CertificationProfile,
    Decision,
    require_single_profile,
)
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

RecertifyVerdict = Literal["replace", "keep", "infeasible"]


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; bare dates become midnight UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc


def _index_unique(rows: list[Any], kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        rid = row.id
        if rid in out:
            raise ValueError(f"duplicate {kind} id: {rid!r}")
        out[rid] = row
    return out


def _matured_audit_outcomes(
    *,
    decisions: list[Decision],
    outcomes: list[OutcomeFeedback],
    as_of: str | None,
    since: str | None,
) -> tuple[list[tuple[str, OutcomeFeedback]], dict[str, Any]]:
    """
    Audit ids with a matured label inside the window, in id order, plus stats.

    Shared by single-model and cascade recertify; prediction lookups and the
    `n_predictions` / `n_audit_labeled` counters are left to the caller.
    """
    dec_by_id = _index_unique(decisions, "decision")
    out_by_id = _index_unique(outcomes, "outcome")

    as_of_dt = _parse_iso(as_of) if as_of is not None else None
    since_dt = _parse_iso(since) if since is not None else None
    if as_of_dt is not None and since_dt is not None and since_dt > as_of_dt:
        raise ValueError(f"--since {since!r} is after --as-of {as_of!r}")

    audit_ids = {d.id for d in decisions if d.audit}
    stats: dict[str, Any] = {
        "n_decisions": len(decisions),
        "n_audit_decisions": len(audit_ids),
        "n_outcomes": len(outcomes),
        "n_audit_labeled": 0,
        "n_skipped_unmatured": 0,
        "n_skipped_before_since": 0,
        "n_skipped_missing_prediction": 0,
        "n_organic_labeled_ignored": 0,
        "as_of": as_of,
        "since": since,
    }

    # Organic labels on non-audit rows are fine to store, never used.
    for oid in out_by_id:
        if oid not in audit_ids and oid in dec_by_id:
            stats["n_organic_labeled_ignored"] += 1

    matured: list[tuple[str, OutcomeFeedback]] = []
    for aid in sorted(audit_ids):
        outcome = out_by_id.get(aid)
        if outcome is None:
            continue
        if as_of_dt is not None:
            if outcome.matured_at is None:
                stats["n_skipped_unmatured"] += 1
                continue
            matured_at = _parse_iso(outcome.matured_at)
            if matured_at > as_of_dt:
                stats["n_skipped_unmatured"] += 1
                continue
            if since_dt is not None and matured_at < since_dt:
                stats["n_skipped_before_since"] += 1
                continue
        elif since_dt is not None:
            if outcome.matured_at is None:
                stats["n_skipped_unmatured"] += 1
                continue
            if _parse_iso(outcome.matured_at) < since_dt:
                stats["n_skipped_before_since"] += 1
                continue
        matured.append((aid, outcome))

    return matured, stats


def _require_choices(choices: list[str] | None) -> list[str]:
    if choices is None:
        raise ValueError(
            "profile has no choices; recertify needs a contract-backed profile "
            "so labels can be checked against the certified choice set"
        )
    return list(choices)


def build_audit_pairs(
    *,
    decisions: list[Decision],
    outcomes: list[OutcomeFeedback],
    predictions: list[DecisionPrediction],
    profile: CertificationProfile,
    as_of: str | None = None,
    since: str | None = None,
) -> tuple[list[tuple[DecisionCase, DecisionPrediction]], dict[str, Any]]:
    """
    Audit-only (case, prediction) pairs for CRC, with join diagnostics.

    Temperature is applied here from the frozen profile so CRC sees the same
    calibrated confidences `decide()` used.
    """
    choices = _require_choices(profile.choices)
    pred_by_id = _index_unique(predictions, "prediction")
    matured, stats = _matured_audit_outcomes(
        decisions=decisions, outcomes=outcomes, as_of=as_of, since=since
    )
    stats["n_predictions"] = len(predictions)

    pairs: list[tuple[DecisionCase, DecisionPrediction]] = []
    for aid, outcome in matured:
        pred = pred_by_id.get(aid)
        if pred is None:
            stats["n_skipped_missing_prediction"] += 1
            continue
        if outcome.label not in choices:
            raise ValueError(
                f"outcome {aid!r}: label {outcome.label!r} not in profile choices {choices}"
            )
        if profile.model and pred.model and pred.model != profile.model:
            raise ValueError(
                f"prediction {aid!r} is from model {pred.model!r}; "
                f"profile certifies {profile.model!r}"
            )

        scored = (
            apply_temperature(pred, profile.temperature)
            if profile.temperature is not None
            else pred
        )
        case = DecisionCase(id=aid, choices=choices, label=outcome.label)
        pairs.append((case, scored))
        stats["n_audit_labeled"] += 1

    return pairs, stats


class RecertifyReport(BaseModel):
    """Outcome of recertifying one profile on its audit sample."""

    verdict: RecertifyVerdict
    """
    `replace` — new certificate is feasible; prefer the new profile.
    `keep` — feasible but not meaningfully different; current profile is fine.
    `infeasible` — audit sample cannot support alpha (too small or too noisy).
    """

    alpha: float
    n_audit_calibration: int
    certificate: CRCResult
    current: dict[str, Any]
    """Snapshot of the live profile fields being compared."""

    proposed: dict[str, Any] | None = None
    """New operating point when feasible."""

    delta: dict[str, float | None] = Field(default_factory=dict)
    join: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    profile: CertificationProfile | None = None
    """New profile when verdict is replace or keep (feasible); None if infeasible."""


def _verdict(
    current: CertificationProfile,
    cert: CRCResult,
    *,
    threshold_tol: float = 1e-4,
) -> RecertifyVerdict:
    if not cert.feasible or cert.threshold is None:
        return "infeasible"
    # A higher threshold is stricter (less automation). Either direction that
    # moves past tol is a real change worth installing.
    if abs(cert.threshold - current.threshold) > threshold_tol:
        return "replace"
    if (
        cert.coverage_calibration is not None
        and current.coverage_calibration is not None
        and abs(cert.coverage_calibration - current.coverage_calibration) > 0.01
    ):
        return "replace"
    return "keep"


def run_recertification(
    profile: CertificationProfile,
    *,
    decisions: list[Decision],
    outcomes: list[OutcomeFeedback],
    predictions: list[DecisionPrediction],
    contract: DecisionContract | None = None,
    as_of: str | None = None,
    since: str | None = None,
    confidence_level: float | None = None,
    loss_bound: float | None = None,
) -> RecertifyReport:
    """
    Re-run CRC on matured audit labels and diff against `profile`.

    `confidence_level` defaults to the profile's own guarantee. Temperature is
    never re-fit. Only `Decision.audit` rows enter the calibration set.
    """
    profile = require_single_profile(profile, what="recertify")
    if contract is not None:
        profile.check_contract(contract)
    elif profile.contract is not None:
        contract = DecisionContract.model_validate(profile.contract)

    pairs, join = build_audit_pairs(
        decisions=decisions,
        outcomes=outcomes,
        predictions=predictions,
        profile=profile,
        as_of=as_of,
        since=since,
    )

    notes: list[str] = [
        "Recertify uses audit=True rows only; organic escalations are ignored.",
        "Temperature is the frozen profile value; it is not re-fit on audit data.",
    ]
    if contract is not None:
        contract.validate_data([c for c, _ in pairs], [p for _, p in pairs])

    auto = auto_set_of(contract) if contract is not None else (
        frozenset(profile.auto_choices) if profile.auto_choices is not None else None
    )
    severity_of = None
    loss_of = None
    quantity = None
    if contract is not None and contract.is_ordinal:
        loss_of = contract.pair_loss
        quantity = ORDINAL_QUANTITY
    elif contract is not None and contract.severity:
        severity_of = contract.miss_severity
    if loss_bound is None:
        if contract is not None:
            loss_bound = contract.loss_bound()
        else:
            raw = (profile.split or {}).get("loss_bound")
            loss_bound = float(raw) if raw is not None else 1.0

    level = profile.confidence_level if confidence_level is None else confidence_level
    # High-probability profiles stay high-probability unless the caller overrides.
    if level is None and profile.guarantee == "high_probability":
        level = profile.confidence_level

    current_snap = {
        "profile_hash": profile.profile_hash,
        "threshold": profile.threshold,
        "crc_bound_value": profile.crc_bound_value,
        "coverage_calibration": profile.coverage_calibration,
        "n_calibration": profile.n_calibration,
        "alpha": profile.alpha,
        "guarantee": profile.guarantee,
        "confidence_level": profile.confidence_level,
        "temperature": profile.temperature,
        "audit_rate": profile.audit_rate,
        "audit_seed": profile.audit_seed,
    }

    if not pairs:
        notes.append(
            "No matured audit labels joined to predictions; cannot recertify. "
            f"audit decisions={join['n_audit_decisions']}, "
            f"labeled={join['n_audit_labeled']}."
        )
        empty = CRCResult(
            alpha=profile.alpha,
            feasible=False,
            n_calibration=0,
            min_n_required=0,
            loss_bound=loss_bound,
            notes=["empty audit calibration set"],
        )
        return RecertifyReport(
            verdict="infeasible",
            alpha=profile.alpha,
            n_audit_calibration=0,
            certificate=empty,
            current=current_snap,
            join=join,
            notes=notes,
        )

    # Keep the kind of guarantee the live profile was certified with.
    if profile.controlled_quantity == SELECTIVE_QUANTITY:
        cert = certify_selective_threshold(
            pairs,
            profile.alpha,
            confidence_level=level if level is not None else 0.9,
            auto_choices=auto,
        )
        severity_of = None
    else:
        cert = certify_threshold(
            pairs,
            profile.alpha,
            loss_bound=loss_bound,
            auto_choices=auto,
            severity_of=severity_of,
            loss_of=loss_of,
            controlled_quantity=quantity,
            confidence_level=level,
        )
    notes.extend(cert.notes)

    # Empirical check on the same audit set (not a held-out guarantee).
    held = None
    if cert.feasible and cert.threshold is not None:
        held = evaluate_certified_threshold(
            pairs,
            cert.threshold,
            auto_choices=auto,
            severity_of=severity_of,
            loss_of=loss_of,
        )
        notes.append(
            "joint_error_rate on the audit sample is empirical (same rows that "
            "chose lambda_hat), not a held-out test."
        )

    verdict = _verdict(profile, cert)
    delta: dict[str, float | None] = {
        "threshold": (
            None if cert.threshold is None else cert.threshold - profile.threshold
        ),
        "coverage_calibration": (
            None
            if cert.coverage_calibration is None or profile.coverage_calibration is None
            else cert.coverage_calibration - profile.coverage_calibration
        ),
        "crc_bound_value": (
            None
            if cert.crc_bound_value is None or profile.crc_bound_value is None
            else cert.crc_bound_value - profile.crc_bound_value
        ),
    }

    proposed = None
    new_profile = None
    if cert.feasible and cert.threshold is not None:
        proposed = {
            "threshold": cert.threshold,
            "crc_bound_value": cert.crc_bound_value,
            "coverage_calibration": cert.coverage_calibration,
            "n_calibration": cert.n_calibration,
            "empirical_joint_error": cert.empirical_joint_error,
            "audit_eval": held,
        }
        new_profile = profile.model_copy(
            update={
                "threshold": cert.threshold,
                "guarantee": cert.guarantee,
                "confidence_level": cert.confidence_level,
                "controlled_quantity": cert.controlled_quantity,
                "n_calibration": cert.n_calibration,
                "n_temperature_fit": 0,
                "crc_bound_value": cert.crc_bound_value,
                "coverage_calibration": cert.coverage_calibration,
                "test": held,
                "split": {
                    "protocol": "audit_recertify",
                    "n_audit_calibration": cert.n_calibration,
                    "as_of": as_of,
                    "since": since,
                    "loss_bound": loss_bound,
                    "parent_profile": profile.profile_hash,
                },
            }
        )
        if verdict == "infeasible":
            pass
        elif verdict == "replace":
            notes.append(
                f"New threshold {cert.threshold:.4f} differs from current "
                f"{profile.threshold:.4f}; install the new profile."
            )
        else:
            notes.append(
                "Audit certificate matches the current operating point within "
                "tolerance; keeping the live profile is fine."
            )
    else:
        notes.append(
            f"Audit sample (n={len(pairs)}) cannot certify alpha={profile.alpha:g}; "
            "keep the current profile only as a temporary measure and raise "
            "audit_rate or wait for more labels."
        )

    return RecertifyReport(
        verdict=verdict,
        alpha=profile.alpha,
        n_audit_calibration=len(pairs),
        certificate=cert,
        current=current_snap,
        proposed=proposed,
        delta=delta,
        join=join,
        notes=notes,
        profile=new_profile,
    )


# --- Two-stage cascade ------------------------------------------------------


def build_cascade_audit_pairs(
    *,
    decisions: list[Decision],
    outcomes: list[OutcomeFeedback],
    stage1: list[DecisionPrediction],
    stage2: list[DecisionPrediction],
    profile: CascadeProfile,
    as_of: str | None = None,
    since: str | None = None,
) -> tuple[
    list[tuple[DecisionCase, DecisionPrediction]],
    list[tuple[DecisionCase, DecisionPrediction]],
    dict[str, Any],
]:
    """
    Audit-only pairs for both stages, same case order, frozen T per stage.

    An audit row needs a prediction from *both* stages: `decide_cascade` fetches
    stage 2 on every audit case for exactly this reason. Rows missing either
    are skipped and counted, never imputed.
    """
    choices = _require_choices(profile.choices)
    by1 = _index_unique(stage1, "stage-1 prediction")
    by2 = _index_unique(stage2, "stage-2 prediction")
    matured, stats = _matured_audit_outcomes(
        decisions=decisions, outcomes=outcomes, as_of=as_of, since=since
    )
    stats["n_predictions"] = [len(stage1), len(stage2)]
    spec1, spec2 = profile.stages

    pairs1: list[tuple[DecisionCase, DecisionPrediction]] = []
    pairs2: list[tuple[DecisionCase, DecisionPrediction]] = []
    for aid, outcome in matured:
        p1, p2 = by1.get(aid), by2.get(aid)
        if p1 is None or p2 is None:
            stats["n_skipped_missing_prediction"] += 1
            continue
        if outcome.label not in choices:
            raise ValueError(
                f"outcome {aid!r}: label {outcome.label!r} not in profile choices {choices}"
            )
        for idx, (pred, spec) in enumerate(((p1, spec1), (p2, spec2)), start=1):
            if spec.model and pred.model and pred.model != spec.model:
                raise ValueError(
                    f"prediction {aid!r} is from model {pred.model!r}; "
                    f"cascade stage {idx} certifies {spec.model!r}"
                )
        s1 = apply_temperature(p1, spec1.temperature) if spec1.temperature is not None else p1
        s2 = apply_temperature(p2, spec2.temperature) if spec2.temperature is not None else p2
        case = DecisionCase(id=aid, choices=choices, label=outcome.label)
        pairs1.append((case, s1))
        pairs2.append((case, s2))
        stats["n_audit_labeled"] += 1

    return pairs1, pairs2, stats


class CascadeRecertifyReport(BaseModel):
    """Outcome of recertifying a cascade profile on its audit sample."""

    verdict: RecertifyVerdict
    alpha: float
    n_audit_calibration: int
    certificate: CascadeCertificate | None = None
    """None only when no audit row joined (nothing to test)."""

    current: dict[str, Any]
    proposed: dict[str, Any] | None = None
    delta: dict[str, Any] = Field(default_factory=dict)
    join: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    profile: CascadeProfile | None = None


def _lam_changed(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return (a is None) != (b is None)
    return abs(a - b) > tol


def _cascade_verdict(
    current: CascadeProfile,
    cert: CascadeCertificate,
    *,
    threshold_tol: float = 1e-4,
    coverage_tol: float = 0.01,
) -> RecertifyVerdict:
    """
    Same rule as single-model `_verdict`: any real move of the operating point
    is installed, stricter or looser. A stricter point after drift must replace
    the live one, because the live one may no longer hold alpha.
    """
    if not cert.feasible:
        return "infeasible"
    cur1, cur2 = (s.threshold for s in current.stages)
    if _lam_changed(cert.lambda1, cur1, threshold_tol) or _lam_changed(
        cert.lambda2, cur2, threshold_tol
    ):
        return "replace"
    new_cov = cert.calibration.automation_rate if cert.calibration else None
    if (
        new_cov is not None
        and current.coverage_calibration is not None
        and abs(new_cov - current.coverage_calibration) > coverage_tol
    ):
        return "replace"
    return "keep"


def _cascade_grids(profile: CascadeProfile, notes: list[str]) -> tuple[list[float], list[float]]:
    """
    Pre-specified grids for the audit LTT: the profile's own, plus its current
    thresholds. Neither depends on audit data, so the hypotheses stay fixed.
    """
    g1 = list(profile.lambda1_grid or [])
    g2 = list(profile.lambda2_grid or [])
    if not g1 or not g2:
        notes.append(
            "Profile predates stored grids; using an even grid on (0, 1) plus the "
            "current thresholds. Recertify once to store grids on the new profile."
        )
        even = quantile_grid([], DEFAULT_GRID_SIZE)
        g1 = g1 or even
        g2 = g2 or even
    cur1, cur2 = (s.threshold for s in profile.stages)
    if cur1 is not None:
        g1 = sorted({*g1, float(cur1)})
    if cur2 is not None:
        g2 = sorted({*g2, float(cur2)})
    return g1, g2


def run_cascade_recertification(
    profile: CascadeProfile,
    *,
    decisions: list[Decision],
    outcomes: list[OutcomeFeedback],
    stage1: list[DecisionPrediction],
    stage2: list[DecisionPrediction],
    contract: DecisionContract | None = None,
    as_of: str | None = None,
    since: str | None = None,
    confidence_level: float | None = None,
) -> CascadeRecertifyReport:
    """
    Re-run the cascade Learn-Then-Test on matured audit labels only.

    T1 / T2 are frozen from the profile, the grids are the profile's own (fixed
    before any audit label), and organic escalations never enter. The verdict
    follows single-model recertify: infeasible, else replace on any real move
    of (lambda1, lambda2) or calibration automation, else keep.
    """
    if contract is not None:
        profile.check_contract(contract)
    elif profile.contract is not None:
        contract = DecisionContract.model_validate(profile.contract)
    if contract is not None and contract.severity:
        raise ValueError("cascade recertify needs 0-1 loss; contract severity is not supported")
    if contract is not None and contract.is_ordinal:
        raise ValueError(
            "cascade recertify needs equality 0-1 loss; ordinal under-rate "
            "loss is not supported"
        )

    pairs1, pairs2, join = build_cascade_audit_pairs(
        decisions=decisions,
        outcomes=outcomes,
        stage1=stage1,
        stage2=stage2,
        profile=profile,
        as_of=as_of,
        since=since,
    )
    notes: list[str] = [
        "Recertify uses audit=True rows only; organic escalations are ignored.",
        "T1 / T2 are the frozen profile values; they are not re-fit on audit data.",
    ]
    level = profile.confidence_level if confidence_level is None else confidence_level
    s1_spec, s2_spec = profile.stages
    current = {
        "profile_hash": profile.profile_hash,
        "lambda1": s1_spec.threshold,
        "lambda2": s2_spec.threshold,
        "upper_bound": profile.crc_bound_value,
        "coverage_calibration": profile.coverage_calibration,
        "n_calibration": profile.n_calibration,
        "alpha": profile.alpha,
        "confidence_level": profile.confidence_level,
        "temperatures": [s1_spec.temperature, s2_spec.temperature],
        "audit_rate": profile.audit_rate,
        "audit_seed": profile.audit_seed,
    }

    if not pairs1:
        notes.append(
            "No matured audit labels joined to both stages' predictions; cannot "
            f"recertify. audit decisions={join['n_audit_decisions']}, "
            f"missing prediction={join['n_skipped_missing_prediction']}."
        )
        return CascadeRecertifyReport(
            verdict="infeasible",
            alpha=profile.alpha,
            n_audit_calibration=0,
            current=current,
            join=join,
            notes=notes,
        )

    if contract is not None:
        cases = [c for c, _ in pairs1]
        contract.validate_data(cases, [p for _, p in pairs1])
        contract.validate_data(cases, [p for _, p in pairs2])
    auto = auto_set_of(contract) if contract is not None else (
        frozenset(profile.auto_choices) if profile.auto_choices is not None else None
    )

    priced = profile.stage_costs is not None and contract is not None and bool(contract.costs)
    path_costs = dict(contract.costs) if priced and contract is not None else None
    stage_costs = (
        (float(profile.stage_costs[0]), float(profile.stage_costs[1]))
        if priced and profile.stage_costs is not None
        else None
    )
    esc = None
    if priced and contract is not None:
        esc = np.array([escalate_cost(contract, c) for c, _ in pairs1], dtype=np.float64)

    g1, g2 = _cascade_grids(profile, notes)
    s1 = stage_arrays_from_pairs(pairs1, auto_choices=auto, path_costs=path_costs)
    s2 = stage_arrays_from_pairs(pairs2, auto_choices=auto, path_costs=path_costs)
    cert = certify_cascade(
        s1,
        s2,
        profile.alpha,
        lambda1_grid=g1,
        lambda2_grid=g2,
        confidence_level=level,
        stage_costs=stage_costs,
        escalate_costs=esc,
    )
    notes.extend(cert.notes)

    verdict = _cascade_verdict(profile, cert)
    proposed = None
    new_profile = None
    delta: dict[str, Any] = {}
    if cert.feasible and cert.calibration is not None:
        cal = cert.calibration
        proposed = {
            "lambda1": cert.lambda1,
            "lambda2": cert.lambda2,
            "upper_bound": cal.upper_bound,
            "coverage_calibration": cal.automation_rate,
            "joint_error_rate": cal.joint_error_rate,
            "n_auto_stage1": cal.n_auto_stage1,
            "n_auto_stage2": cal.n_auto_stage2,
            "n_calibration": cert.n_calibration,
        }
        delta = {
            "lambda1_changed": _lam_changed(cert.lambda1, s1_spec.threshold, 1e-4),
            "lambda2_changed": _lam_changed(cert.lambda2, s2_spec.threshold, 1e-4),
            "coverage_calibration": (
                None
                if profile.coverage_calibration is None
                else cal.automation_rate - profile.coverage_calibration
            ),
            "upper_bound": (
                None
                if cal.upper_bound is None or profile.crc_bound_value is None
                else cal.upper_bound - profile.crc_bound_value
            ),
        }
        new_profile = profile.model_copy(
            update={
                "stages": [
                    s1_spec.model_copy(update={"threshold": cert.lambda1}),
                    s2_spec.model_copy(update={"threshold": cert.lambda2}),
                ],
                "confidence_level": cert.confidence_level,
                "n_calibration": cert.n_calibration,
                "n_rows": cert.n_rows,
                "per_row_level": cert.per_row_level,
                "objective": cert.objective,
                "lambda1_grid": list(cert.lambda1_grid),
                "lambda2_grid": list(cert.lambda2_grid),
                "crc_bound_value": cal.upper_bound,
                "coverage_calibration": cal.automation_rate,
                "test": cal.model_dump(mode="json"),
                "split": {
                    "protocol": "audit_recertify_cascade",
                    "n_audit_calibration": cert.n_calibration,
                    "as_of": as_of,
                    "since": since,
                    "parent_profile": profile.profile_hash,
                },
            }
        )
        notes.append(
            "Calibration numbers are on the audit sample that chose the point, "
            "not a held-out test."
        )
        if verdict == "replace":
            notes.append("Operating point moved; install the new profile.")
        else:
            notes.append(
                "Audit certificate matches the current operating point within "
                "tolerance; keeping the live profile is fine."
            )
    else:
        if len(pairs1) < cert.min_n_required:
            why = f"needs >= {cert.min_n_required} even with zero errors"
        else:
            why = "no grid point keeps the upper bound <= alpha on this sample"
        notes.append(
            f"Audit sample (n={len(pairs1)}) cannot certify alpha={profile.alpha:g} at "
            f"{level:.0%}: {why}. The live profile may no longer hold alpha; keep it "
            "only as a temporary measure, raise audit_rate or wait for more labels."
        )

    return CascadeRecertifyReport(
        verdict=verdict,
        alpha=profile.alpha,
        n_audit_calibration=len(pairs1),
        certificate=cert,
        current=current,
        proposed=proposed,
        delta=delta,
        join=join,
        notes=notes,
        profile=new_profile,
    )
