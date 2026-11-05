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

**Audit sample.** Once the cheap path is live, auto-accepted cases never get
the expensive verdict, so organic labels come only from low-confidence
escalations — a biased sample that cannot recertify. With `audit_rate = eps`
each case is drawn into an audit sample with probability eps, from a hash of
`(audit_seed, id)`: independent of the model's score, reproducible, and
stateless across processes. Every drawn case is marked `audit=True`, including
those that would escalate anyway, so the audit set is a uniform sample of
traffic and a valid calibration set as-is. A drawn case the rule would AUTO is
escalated with `reason="audit"`. Forcing escalations only removes automated
errors, so the certified bound still holds; the cost is ~eps of the savings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from decisionops.__version__ import __version__
from decisionops.calibration.temperature import apply_temperature
from decisionops.conformal.cascade_certify import CascadeCertificationReport
from decisionops.conformal.cascade_risk import CASCADE_QUANTITY, CascadeCertificate
from decisionops.conformal.certify import CertificationReport
from decisionops.conformal.risk_control import GuaranteeKind
from decisionops.contract import DecisionContract
from decisionops.schemas import DecisionPrediction

PROFILE_SCHEMA_VERSION = 1

Action = Literal["auto", "escalate"]
Reason = Literal[
    "certified", "below_threshold", "escalate_choice", "outside_contract", "audit"
]

_AUDIT_SCALE = float(2**64)


def audit_draw(audit_seed: int, case_id: str) -> float:
    """Uniform [0, 1) draw for one case, from `(audit_seed, id)` only."""
    digest = hashlib.sha256(f"{audit_seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _AUDIT_SCALE


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

    audit_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Share of traffic drawn into the random audit sample (forced to the expensive path)."""

    audit_seed: int = 0
    """Salt for the audit draw; with the case id it fully determines membership."""

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
                "(choices, auto set, severity, descriptions or instructions "
                "changed); recertify"
            )


class CascadeStage(BaseModel):
    """One stage of a two-model cascade profile."""

    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    threshold: float | None = None
    """Certified lambda for this stage; None means the stage is off (never called)."""


class CascadeProfile(BaseModel):
    """Two-stage cascade certificate, ready for runtime use."""

    schema_version: int = PROFILE_SCHEMA_VERSION
    kind: Literal["cascade"] = "cascade"
    alpha: float
    guarantee: Literal["high_probability"] = "high_probability"
    confidence_level: float
    controlled_quantity: str = CASCADE_QUANTITY
    stages: list[CascadeStage]
    """Exactly two stages; `threshold is None` disables that stage."""

    choices: list[str] | None = None
    auto_choices: list[str] | None = None
    contract_hash: str | None = None
    contract: dict[str, Any] | None = None
    audit_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    audit_seed: int = 0
    n_calibration: int
    n_rows: int | None = None
    per_row_level: float | None = None
    objective: str = "automation"
    stage_costs: list[float] | None = None
    """Per-call cost of (M1, M2) when the cost objective was used; reused by recertify."""

    lambda1_grid: list[float] | None = None
    lambda2_grid: list[float] | None = None
    """
    Threshold grids the certificate was tested on, fixed before its labels were
    seen. Recertify reuses them so its hypotheses do not depend on audit labels.
    """

    crc_bound_value: float | None = None
    coverage_calibration: float | None = None
    test: dict[str, Any] | None = None
    split: dict[str, Any] = Field(default_factory=dict)
    decisionops_version: str = __version__

    @model_validator(mode="after")
    def _two_stages(self) -> CascadeProfile:
        if len(self.stages) != 2:
            raise ValueError(f"cascade profile needs exactly 2 stages, got {len(self.stages)}")
        return self

    @property
    def profile_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def check_contract(self, contract: DecisionContract | None) -> None:
        if contract is None:
            return
        if self.contract_hash is None:
            raise ProfileMismatchError(
                f"profile was certified without a contract; got contract {contract.name!r}"
            )
        if contract.fingerprint() != self.contract_hash:
            raise ProfileMismatchError(
                f"contract {contract.name!r} differs from the certified one "
                "(choices, auto set, severity, descriptions or instructions "
                "changed); recertify"
            )


AnyProfile = CertificationProfile | CascadeProfile


def require_single_profile(profile: AnyProfile, *, what: str) -> CertificationProfile:
    """Reject a cascade profile with a clear error; used by recertify, drift, decide."""
    if isinstance(profile, CascadeProfile):
        raise ValueError(
            f"{what} does not support cascade profiles yet; "
            "route with decide_cascade() and recertify with run_cascade_recertification()"
        )
    return profile


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
    stage: int | None = None
    """1 or 2 when that cascade stage would auto-execute; None for a single-model profile."""

    audit: bool = False
    """
    Drawn into the random audit sample. True also for cases that escalated for
    their own reason; recertify on exactly the `audit=True` cases.
    """

    profile: str
    """`CertificationProfile.profile_hash`."""


def profile_from_report(
    report: CertificationReport,
    alpha: float,
    *,
    audit_rate: float = 0.0,
    audit_seed: int = 0,
) -> CertificationProfile:
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
        audit_rate=audit_rate,
        audit_seed=audit_seed,
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
    require_single_profile(profile, what="decide()")
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

    audit = (
        profile.audit_rate > 0.0
        and audit_draw(profile.audit_seed, prediction.id) < profile.audit_rate
    )
    if audit and reason == "certified":
        reason = "audit"

    return Decision(
        id=prediction.id,
        action="auto" if reason == "certified" else "escalate",
        choice=choice,
        confidence=confidence,
        threshold=profile.threshold,
        alpha=profile.alpha,
        reason=reason,
        audit=audit,
        profile=tag,
    )


def _apply_stage_temperature(
    prediction: DecisionPrediction, temperature: float | None
) -> DecisionPrediction:
    if temperature is None:
        return prediction
    return apply_temperature(prediction, temperature)


def _check_stage_model(prediction: DecisionPrediction, stage: CascadeStage, index: int) -> None:
    if stage.model and prediction.model and prediction.model != stage.model:
        raise ProfileMismatchError(
            f"prediction {prediction.id!r} is from model {prediction.model!r}; "
            f"cascade stage {index} certifies {stage.model!r}"
        )


def _stage_reason(
    scored: DecisionPrediction,
    profile: CascadeProfile,
    threshold: float,
) -> Reason:
    if profile.choices is not None and scored.prediction not in profile.choices:
        return "outside_contract"
    if profile.auto_choices is not None and scored.prediction not in profile.auto_choices:
        return "escalate_choice"
    if scored.confidence >= threshold:
        return "certified"
    return "below_threshold"


def cascade_profile_from(
    cert: CascadeCertificate,
    stage1: CascadeStage,
    stage2: CascadeStage,
    *,
    contract: DecisionContract | None = None,
    audit_rate: float = 0.0,
    audit_seed: int = 0,
    test: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    stage_costs: tuple[float, float] | list[float] | None = None,
) -> CascadeProfile:
    """Freeze a feasible cascade certificate into a runtime profile."""
    if not cert.feasible:
        raise ValueError("no certified cascade point; cannot write a profile")
    return CascadeProfile(
        alpha=cert.alpha,
        confidence_level=cert.confidence_level,
        controlled_quantity=cert.controlled_quantity,
        stages=[
            stage1.model_copy(update={"threshold": cert.lambda1}),
            stage2.model_copy(update={"threshold": cert.lambda2}),
        ],
        choices=list(contract.choices) if contract is not None else None,
        auto_choices=list(contract.auto_choices) if contract is not None else None,
        contract_hash=contract.fingerprint() if contract is not None else None,
        contract=contract.summary() if contract is not None else None,
        audit_rate=audit_rate,
        audit_seed=audit_seed,
        n_calibration=cert.n_calibration,
        n_rows=cert.n_rows,
        per_row_level=cert.per_row_level,
        objective=cert.objective,
        stage_costs=list(stage_costs) if stage_costs is not None else None,
        lambda1_grid=list(cert.lambda1_grid),
        lambda2_grid=list(cert.lambda2_grid),
        crc_bound_value=cert.calibration.upper_bound if cert.calibration else None,
        coverage_calibration=(
            cert.calibration.automation_rate if cert.calibration else None
        ),
        test=test,
        split=split or {},
    )


def decide_cascade(
    stage1: DecisionPrediction,
    profile: CascadeProfile,
    *,
    stage2: DecisionPrediction | None = None,
    get_stage2: Callable[[], DecisionPrediction] | None = None,
    profile_hash: str | None = None,
) -> Decision:
    """
    Route one case through a two-stage cascade.

    Stage 2 is fetched only when stage 1 does not auto-execute, or when the
    case is drawn into the audit sample (so recertify later has both scores).
    Pass `get_stage2` to avoid calling M2 on cheap autos; `stage2` is used
    when already computed (offline JSONL).
    """
    spec1, spec2 = profile.stages
    tag = profile_hash or profile.profile_hash
    fetched2: DecisionPrediction | None = None

    def need_stage2() -> DecisionPrediction:
        nonlocal fetched2
        if fetched2 is not None:
            return fetched2
        if stage2 is not None:
            fetched2 = stage2
            return fetched2
        if get_stage2 is not None:
            fetched2 = get_stage2()
            return fetched2
        raise ValueError(
            f"prediction {stage1.id!r} needs a stage-2 prediction "
            "(stage 1 did not auto-execute, or the case is in the audit sample)"
        )

    scored = stage1
    threshold = spec1.threshold if spec1.threshold is not None else 0.0
    reason: Reason = "below_threshold"
    would_stage: int | None = None

    if spec1.threshold is not None:
        _check_stage_model(stage1, spec1, 1)
        scored = _apply_stage_temperature(stage1, spec1.temperature)
        threshold = spec1.threshold
        reason = _stage_reason(scored, profile, spec1.threshold)
        if reason == "certified":
            would_stage = 1

    if would_stage is None and spec2.threshold is not None:
        raw2 = need_stage2()
        _check_stage_model(raw2, spec2, 2)
        scored = _apply_stage_temperature(raw2, spec2.temperature)
        threshold = spec2.threshold
        reason = _stage_reason(scored, profile, spec2.threshold)
        if reason == "certified":
            would_stage = 2

    audit = (
        profile.audit_rate > 0.0
        and audit_draw(profile.audit_seed, stage1.id) < profile.audit_rate
    )
    if audit and spec2.threshold is not None:
        raw2 = need_stage2()
        _check_stage_model(raw2, spec2, 2)
        _apply_stage_temperature(raw2, spec2.temperature)
    if audit and reason == "certified":
        reason = "audit"

    return Decision(
        id=stage1.id,
        action="auto" if reason == "certified" else "escalate",
        choice=scored.prediction,
        confidence=scored.confidence,
        threshold=float(threshold),
        alpha=profile.alpha,
        reason=reason,
        stage=would_stage,
        audit=audit,
        profile=tag,
    )


def decide_cascade_many(
    stage1: list[DecisionPrediction],
    profile: CascadeProfile,
    stage2: list[DecisionPrediction],
) -> list[Decision]:
    """Route many cases; stage-2 rows are looked up by id only when needed."""
    by_id = {p.id: p for p in stage2}
    if len(by_id) != len(stage2):
        raise ValueError("duplicate stage-2 prediction ids")
    tag = profile.profile_hash
    return [
        decide_cascade(p1, profile, stage2=by_id.get(p1.id), profile_hash=tag)
        for p1 in stage1
    ]


def cascade_profile_from_report(
    report: CascadeCertificationReport,
    *,
    audit_rate: float = 0.0,
    audit_seed: int = 0,
) -> CascadeProfile:
    """Freeze a `CascadeCertificationReport` into a runtime profile."""
    contract_raw = (report.metadata or {}).get("contract")
    contract = DecisionContract.model_validate(contract_raw) if contract_raw else None
    s1, s2 = report.stages
    test = report.test.model_dump(mode="json") if report.test is not None else None
    return cascade_profile_from(
        report.certificate,
        CascadeStage(provider=s1.provider, model=s1.model, temperature=s1.temperature),
        CascadeStage(provider=s2.provider, model=s2.model, temperature=s2.temperature),
        contract=contract,
        audit_rate=audit_rate,
        audit_seed=audit_seed,
        test=test,
        split=report.split.model_dump(mode="json"),
        stage_costs=(report.metadata or {}).get("stage_costs"),
    )


def decide_many(
    predictions: list[DecisionPrediction], profile: CertificationProfile
) -> list[Decision]:
    tag = profile.profile_hash
    return [decide(p, profile, profile_hash=tag) for p in predictions]


def save_profile(profile: AnyProfile, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False)
    out.write_text(payload + "\n", encoding="utf-8")
    return out


def load_profile(path: str | Path) -> AnyProfile:
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
    kind = raw.get("kind")
    if kind == "cascade":
        return CascadeProfile.model_validate(raw)
    if kind not in (None, "single"):
        raise ValueError(f"profile {p} has unknown kind={kind!r}")
    return CertificationProfile.model_validate(raw)
