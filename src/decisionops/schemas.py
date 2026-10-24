"""Core data schemas for datasets, predictions, and eval results."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class DecisionCase(BaseModel):
    """One labeled decision example."""

    id: str
    state: str = ""
    question: str = ""
    choices: list[str] = Field(min_length=2)
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def label_must_be_in_choices(self) -> DecisionCase:
        if self.label not in self.choices:
            raise ValueError(
                f"label {self.label!r} not in choices {self.choices!r} (id={self.id!r})"
            )
        return self


class DecisionPrediction(BaseModel):
    """Model output for one case, with a full probability distribution."""

    id: str
    prediction: str
    probabilities: dict[str, float]
    latency_ms: float | None = None
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("probabilities")
    @classmethod
    def probabilities_must_be_valid(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) < 2:
            raise ValueError("probabilities must contain at least 2 classes")
        # NaN slips past `p < 0` and would make confidence NaN, silently
        # escalating every case; inf normalises to NaN.
        if any(not math.isfinite(p) for p in value.values()):
            raise ValueError("probabilities must be finite")
        if any(p < 0.0 for p in value.values()):
            raise ValueError("probabilities must be non-negative")
        total = float(sum(value.values()))
        if total <= 0.0:
            raise ValueError("probabilities must sum to a positive value")
        # Allow mild float drift; renormalize only when clearly off.
        if abs(total - 1.0) > 1e-3:
            value = {k: float(v) / total for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def prediction_must_match_probs(self) -> DecisionPrediction:
        if self.prediction not in self.probabilities:
            raise ValueError(
                f"prediction {self.prediction!r} not in probabilities "
                f"{list(self.probabilities)!r} (id={self.id!r})"
            )
        return self

    @property
    def confidence(self) -> float:
        """Max-class probability (raw confidence)."""
        return float(max(self.probabilities.values()))

    def probability_for(self, label: str, default: float = 0.0) -> float:
        return float(self.probabilities.get(label, default))


class ThresholdRecommendation(BaseModel):
    """Selective-prediction operating point for a target risk."""

    target_risk: float
    threshold: float
    coverage: float
    selective_accuracy: float
    selective_risk: float
    n_accepted: int
    n_total: int


class LatencyStats(BaseModel):
    count: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    mean_ms: float | None = None


class EvalResult(BaseModel):
    """Aggregate evaluation report for one model on one dataset."""

    n_samples: int
    accuracy: float
    ece: float
    brier: float
    nll: float
    latency: LatencyStats = Field(default_factory=LatencyStats)
    thresholds: list[ThresholdRecommendation] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SplitInfo(BaseModel):
    """How calibration / test partitions were formed."""

    seed: int
    cal_ratio: float
    n_total: int
    n_calibration: int
    n_test: int


class ConditionResult(BaseModel):
    """Metrics for one probability condition (raw or calibrated)."""

    name: str
    temperature: float | None = None
    n_samples: int
    accuracy: float
    ece: float
    brier: float
    nll: float
    thresholds: list[ThresholdRecommendation] = Field(default_factory=list)
    # Thresholds chosen on calibration; metrics below are on test.
    cal_thresholds: list[ThresholdRecommendation] = Field(default_factory=list)
    reliability_bins: list[dict[str, Any]] = Field(default_factory=list)
    risk_coverage: list[dict[str, Any]] = Field(default_factory=list)


class GateVerdict(BaseModel):
    """
    Go / Watch / No-Go for the technical gate, decided by a CRC certificate.

    - no-go: no certified threshold at `verdict_alpha`
    - watch: certified, but test coverage below `min_coverage`
    - go:    certified and test coverage >= `min_coverage`

    ECE and empirical coverage are reported for context only; they do not pick
    the verdict. Good calibration does not mean safe automation (findings §3).
    """

    recommendation: str  # "go" | "watch" | "no-go"
    verdict_alpha: float = 0.05
    """Target joint automated error rate the verdict is certified at."""

    min_coverage: float = 0.30
    """Certified test coverage needed for `go`. A policy choice, not theory."""

    certified_feasible: bool = False
    certified_threshold: float | None = None
    certified_coverage: float | None = None
    """Share of test traffic auto-accepted at the certified threshold."""

    certified_joint_error: float | None = None
    # Context only — not used in the decision.
    ece_improved: bool = False
    ece_relative_drop: float | None = None
    coverage_at_1pct: float | None = None
    coverage_at_5pct: float | None = None
    """Empirical (not certified) coverage at 1% / 5%, focus condition."""

    notes: list[str] = Field(default_factory=list)


class GateReport(BaseModel):
    """Technical-gate report: split protocol + raw vs calibrated on test."""

    split: SplitInfo
    provider: str | None = None
    model: str | None = None
    target_risks: list[float]
    n_bins: int = 10
    temperature: float | None = None
    temperature_fit: dict[str, Any] = Field(default_factory=dict)
    isotonic_fit: dict[str, Any] = Field(default_factory=dict)
    conditions: list[ConditionResult]
    verdict: GateVerdict
    latency: LatencyStats = Field(default_factory=LatencyStats)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonEntry(BaseModel):
    """One model's gate outcome inside a comparison."""

    name: str
    provider: str | None = None
    model: str | None = None
    n_test: int
    accuracy: float
    ece_raw: float
    ece_calibrated: float | None = None
    brier_raw: float
    brier_calibrated: float | None = None
    nll_raw: float
    nll_calibrated: float | None = None
    temperature: float | None = None
    # Empirical coverage at each target risk, from the calibrated condition when present.
    coverage_by_risk: dict[str, float] = Field(default_factory=dict)
    selective_risk_by_risk: dict[str, float] = Field(default_factory=dict)
    threshold_by_risk: dict[str, float] = Field(default_factory=dict)
    # CRC certificate at the same alphas. Coverage/joint error are measured on test;
    # feasibility is decided on the CRC calibration set, not on that draw.
    n_crc_calibration: int = 0
    n_temperature_fit: int = 0
    certified_feasible_by_risk: dict[str, bool] = Field(default_factory=dict)
    certified_threshold_by_risk: dict[str, float] = Field(default_factory=dict)
    certified_coverage_by_risk: dict[str, float] = Field(default_factory=dict)
    certified_joint_error_by_risk: dict[str, float] = Field(default_factory=dict)
    latency_p95_ms: float | None = None
    verdict: str


class ComparisonReport(BaseModel):
    """Side-by-side comparison of several models on one dataset."""

    dataset: str
    split: SplitInfo
    target_risks: list[float]
    entries: list[ComparisonEntry]
    best_by_coverage_at: dict[str, str] = Field(default_factory=dict)
    """Winner per alpha: max certified coverage among feasible CRC certificates."""
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
