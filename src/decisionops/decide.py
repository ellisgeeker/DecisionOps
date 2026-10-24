"""
Runtime decisions from an offline CRC certificate.

`certify` produces a threshold lambda_hat on (temperature-scaled) confidence.
A `CertificationProfile` freezes one alpha of that report together with
everything the threshold depends on, and `decide()` applies it to a single new
prediction:

    AUTO      prediction is an automatic choice AND calibrated conf >= lambda_hat
    ESCALATE  anything else

This is exactly the acceptance rule CRC certified, so the guarantee carries
over: the joint automated miss rate on exchangeable traffic is <= alpha in
expectation. It holds only for the model, contract input and calibrator that
were certified — the profile records them and `decide()` refuses a mismatch
rather than silently routing under a certificate that no longer applies.

Failure is one-sided on purpose. Anything the rule cannot vouch for (an
escalation choice, a choice outside the contract, low confidence) escalates;
escalating costs money, never safety.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from decisionops.__version__ import __version__
from decisionops.calibration.temperature import apply_temperature
from decisionops.conformal.certify import CertificationReport
from decisionops.conformal.risk_control import GuaranteeKind
from decisionops.contract import DecisionContract
from decisionops.schemas import DecisionPrediction

PROFILE_SCHEMA_VERSION = 1

Action = Literal["auto", "escalate"]
Reason = Literal["certified", "below_threshold", "escalate_choice", "outside_contract"]


class ProfileMismatchError(ValueError):
    """The certificate does not apply to this model or contract."""


class CertificationProfile(BaseModel):
    """One certified operating point, ready for runtime use."""

    schema_version: int = PROFILE_SCHEMA_VERSION
    alpha: float
    guarantee: GuaranteeKind = "expectation"
    """`expectation` (CRC) or `high_probability` (at `confidence_level`)."""

    confidence_level: float | None = None
    controlled_quantity: str = "joint_automated_error_rate"
    threshold: float
    """lambda_hat, on calibrated confidence when `temperature` is set."""

    temperature: float | None = None
    """Applied to raw probabilities before comparing to `threshold`."""

    provider: str | None = None
    model: str | None = None
    choices: list[str] | None = None
    """Contract choice set; None when certified without a contract."""

    auto_choices: list[str] | None = None
    """None means every choice is automatic (certified without a contract)."""

    contract_hash: str | None = None
    contract: dict[str, Any] | None = None

    # Evidence behind the certificate; informational at runtime.
    n_calibration: int
    n_temperature_fit: int = 0
    crc_bound_value: float | None = None
    coverage_calibration: float | None = None
    test: dict[str, float | int] | None = None
    split: dict[str, Any] = Field(default_factory=dict)
    decisionops_version: str = __version__

    @property
    def profile_hash(self) -> str:
        """Stable id for this profile, e.g. to tag decisions in logs."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def check_contract(self, contract: DecisionContract | None) -> None:
        """Raise unless `contract` is the one this profile was certified with."""
        if contract is None:
            return
        if self.contract_hash is None:
            raise ProfileMismatchError(
                f"profile was certified without a contract; got contract {contract.name!r}"
            )
        if contract.fingerprint() != self.contract_hash:
            raise ProfileMismatchError(
                f"contract {contract.name!r} differs from the certified one "
                "(choices, auto set, descriptions or instructions changed); recertify"
            )


class Decision(BaseModel):
    """Routing outcome for one prediction."""

    id: str
    action: Action
    choice: str
    """The model's predicted choice (executed when AUTO)."""

    confidence: float
    """Confidence compared to the threshold (calibrated when the profile has T)."""

    threshold: float
    alpha: float
    reason: Reason
    profile: str
    """`CertificationProfile.profile_hash`."""


def profile_from_report(report: CertificationReport, alpha: float) -> CertificationProfile:
    """Freeze the certificate at `alpha` from a certification report."""
    level = next(
        (lv for lv in report.levels if abs(lv.certificate.alpha - alpha) < 1e-12), None
    )
    if level is None:
        have = ", ".join(f"{lv.certificate.alpha:g}" for lv in report.levels)
        raise ValueError(f"alpha={alpha:g} not in report (have: {have})")
    cert = level.certificate
    if not cert.feasible or cert.threshold is None:
        feasible = ", ".join(
            f"{lv.certificate.alpha:g}" for lv in report.levels if lv.certificate.feasible
        )
        raise ValueError(
            f"no certified threshold at alpha={alpha:g}; "
            f"feasible levels: {feasible or 'none'}"
        )

    meta = report.metadata or {}
    contract = meta.get("contract")
    return CertificationProfile(
        alpha=cert.alpha,
        guarantee=cert.guarantee,
        confidence_level=cert.confidence_level,
        controlled_quantity=cert.controlled_quantity,
        threshold=cert.threshold,
        temperature=report.temperature,
        provider=report.provider,
        model=report.model,
        choices=list(contract["choices"]) if contract and "choices" in contract else None,
        auto_choices=list(contract["auto_choices"]) if contract else None,
        contract_hash=meta.get("contract_hash"),
        contract=contract,
        n_calibration=cert.n_calibration,
        n_temperature_fit=report.n_temperature_fit,
        crc_bound_value=cert.crc_bound_value,
        coverage_calibration=cert.coverage_calibration,
        test=level.test,
        split={
            **report.split.model_dump(mode="json"),
            "temp_ratio": meta.get("temp_ratio"),
            "loss_bound": meta.get("loss_bound"),
        },
    )


def decide(
    prediction: DecisionPrediction,
    profile: CertificationProfile,
    *,
    profile_hash: str | None = None,
) -> Decision:
    """
    Route one prediction with the certified rule.

    Raises `ProfileMismatchError` when the prediction names a different model
    than the one certified. Pass `profile_hash` to skip rehashing the profile
    on every call in a hot loop.
    """
    if profile.model and prediction.model and prediction.model != profile.model:
        raise ProfileMismatchError(
            f"prediction {prediction.id!r} is from model {prediction.model!r}; "
            f"profile certifies {profile.model!r}"
        )
    tag = profile_hash or profile.profile_hash

    scored = (
        apply_temperature(prediction, profile.temperature)
        if profile.temperature is not None
        else prediction
    )
    confidence = scored.confidence
    choice = scored.prediction

    if profile.choices is not None and choice not in profile.choices:
        reason: Reason = "outside_contract"
    elif profile.auto_choices is not None and choice not in profile.auto_choices:
        reason = "escalate_choice"
    elif confidence >= profile.threshold:
        reason = "certified"
    else:
        reason = "below_threshold"

    return Decision(
        id=prediction.id,
        action="auto" if reason == "certified" else "escalate",
        choice=choice,
        confidence=confidence,
        threshold=profile.threshold,
        alpha=profile.alpha,
        reason=reason,
        profile=tag,
    )


def decide_many(
    predictions: list[DecisionPrediction], profile: CertificationProfile
) -> list[Decision]:
    tag = profile.profile_hash
    return [decide(p, profile, profile_hash=tag) for p in predictions]


def save_profile(profile: CertificationProfile, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False)
    out.write_text(payload + "\n", encoding="utf-8")
    return out


def load_profile(path: str | Path) -> CertificationProfile:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"profile not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in profile {p}: {exc}") from exc
    version = raw.get("schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"profile {p} has schema_version={version!r}; "
            f"this build reads {PROFILE_SCHEMA_VERSION}"
        )
    return CertificationProfile.model_validate(raw)
