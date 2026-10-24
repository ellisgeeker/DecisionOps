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

from pydantic import BaseModel, Field

from decisionops.calibration.temperature import apply_temperature
from decisionops.conformal.risk_control import (
    CRCResult,
    certify_threshold,
    evaluate_certified_threshold,
)
from decisionops.contract import DecisionContract, auto_set_of
from decisionops.decide import CertificationProfile, Decision
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
    if profile.choices is None:
        raise ValueError(
            "profile has no choices; recertify needs a contract-backed profile "
            "so labels can be checked against the certified choice set"
        )

    dec_by_id = _index_unique(decisions, "decision")
    out_by_id = _index_unique(outcomes, "outcome")
    pred_by_id = _index_unique(predictions, "prediction")

    as_of_dt = _parse_iso(as_of) if as_of is not None else None
    since_dt = _parse_iso(since) if since is not None else None
    if as_of_dt is not None and since_dt is not None and since_dt > as_of_dt:
        raise ValueError(f"--since {since!r} is after --as-of {as_of!r}")

    audit_ids = {d.id for d in decisions if d.audit}
    stats: dict[str, Any] = {
        "n_decisions": len(decisions),
        "n_audit_decisions": len(audit_ids),
        "n_outcomes": len(outcomes),
        "n_predictions": len(predictions),
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

    pairs: list[tuple[DecisionCase, DecisionPrediction]] = []
    choices = list(profile.choices)
    for aid in sorted(audit_ids):
        outcome = out_by_id.get(aid)
        if outcome is None:
            continue
        if as_of_dt is not None:
            if outcome.matured_at is None:
                stats["n_skipped_unmatured"] += 1
                continue
            matured = _parse_iso(outcome.matured_at)
            if matured > as_of_dt:
                stats["n_skipped_unmatured"] += 1
                continue
            if since_dt is not None and matured < since_dt:
                stats["n_skipped_before_since"] += 1
                continue
        elif since_dt is not None:
            if outcome.matured_at is None:
                stats["n_skipped_unmatured"] += 1
                continue
            if _parse_iso(outcome.matured_at) < since_dt:
                stats["n_skipped_before_since"] += 1
                continue

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
    severity_of = contract.miss_severity if contract is not None else None
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

    cert = certify_threshold(
        pairs,
        profile.alpha,
        loss_bound=loss_bound,
        auto_choices=auto,
        severity_of=severity_of,
        confidence_level=level,
    )
    notes.extend(cert.notes)

    # Empirical check on the same audit set (not a held-out guarantee).
    held = None
    if cert.feasible and cert.threshold is not None:
        held = evaluate_certified_threshold(
            pairs, cert.threshold, auto_choices=auto, severity_of=severity_of
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
