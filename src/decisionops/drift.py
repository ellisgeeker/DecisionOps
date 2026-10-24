"""
Label-free drift monitor for a live certification profile.

Between recertifications, audit labels arrive slowly. This module watches the
*behaviour* of `decide()` — certified rate and confidence distribution — against
a baseline window, without needing outcomes. A shift is an alarm to raise
`audit_rate` or run `recertify`, not a new certificate.

**It cannot see concept drift.** If the model gets worse at the same
confidence, the score distribution does not move and this monitor says `ok`
(findings §10: `ok` on 190/200 reps while true risk doubled). So `ok` never
means the certificate still holds. Recertify on a fixed schedule from audit
labels; use this monitor only to recertify *earlier*.

**Certified rate vs automation rate.** `Decision.audit` forces some would-be
AUTOs to escalate, so raw `action==auto` is pulled down by ~audit_rate.
`reason in {certified, audit}` recovers the CRC acceptance rate and is
comparable to `CertificationProfile.coverage_calibration`. Both are reported;
verdicts key off the certified rate and confidence shift.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp

from decisionops.decide import CertificationProfile, Decision

DriftVerdict = Literal["ok", "watch", "recertify_suggested"]

# Population Stability Index bands (industry rule of thumb for scorecards).
_PSI_WATCH = 0.10
_PSI_ALERT = 0.25
# Absolute gap in certified-rate (fraction of traffic).
_RATE_WATCH = 0.05
_RATE_ALERT = 0.10
# Two-sample KS on confidence: small p => distributions differ.
_KS_WATCH = 0.05
_KS_ALERT = 0.01

_CERTIFIED_REASONS = frozenset({"certified", "audit"})


class WindowStats(BaseModel):
    """Behaviour summary for one decisions window."""

    n: int
    certified_rate: float
    """Share with reason in {certified, audit} — CRC acceptance, audit-invariant."""

    auto_rate: float
    """Share with action==auto (after audit forced escalations)."""

    audit_rate_observed: float
    mean_confidence: float
    conf_p50: float
    conf_p90: float
    reason_counts: dict[str, int] = Field(default_factory=dict)
    choice_rates: dict[str, float] = Field(default_factory=dict)


class DriftReport(BaseModel):
    """Label-free comparison of a live window to a baseline."""

    verdict: DriftVerdict
    profile_hash: str | None = None
    baseline: WindowStats
    live: WindowStats
    delta_certified_rate: float
    delta_auto_rate: float
    psi_confidence: float | None = None
    """Population Stability Index on confidence bins; None if baseline too thin."""

    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    expected_certified_rate: float | None = None
    """Profile coverage_calibration when present; context only."""

    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _confidence_array(decisions: list[Decision]) -> np.ndarray:
    return np.asarray([d.confidence for d in decisions], dtype=np.float64)


def window_stats(decisions: list[Decision]) -> WindowStats:
    if not decisions:
        raise ValueError("cannot summarise an empty decisions window")
    n = len(decisions)
    certified = sum(1 for d in decisions if d.reason in _CERTIFIED_REASONS)
    auto = sum(1 for d in decisions if d.action == "auto")
    audit = sum(1 for d in decisions if d.audit)
    conf = _confidence_array(decisions)
    reasons: dict[str, int] = {}
    choices: dict[str, int] = {}
    for d in decisions:
        reasons[d.reason] = reasons.get(d.reason, 0) + 1
        choices[d.choice] = choices.get(d.choice, 0) + 1
    choice_rates = {k: v / n for k, v in sorted(choices.items())}
    return WindowStats(
        n=n,
        certified_rate=certified / n,
        auto_rate=auto / n,
        audit_rate_observed=audit / n,
        mean_confidence=float(conf.mean()),
        conf_p50=float(np.quantile(conf, 0.50)),
        conf_p90=float(np.quantile(conf, 0.90)),
        reason_counts=dict(sorted(reasons.items())),
        choice_rates=choice_rates,
    )


def population_stability_index(
    baseline: np.ndarray,
    live: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """
    PSI between two 1-d samples using equal-width bins on the pooled range.

    Empty bins get a floor of 0.5 counts so the log is defined. Same convention
    as common credit-score monitoring; not a formal hypothesis test.
    """
    if baseline.size < 2 or live.size < 2:
        raise ValueError("PSI needs at least 2 samples in each window")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    lo = float(min(baseline.min(), live.min()))
    hi = float(max(baseline.max(), live.max()))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("confidence values must be finite for PSI")
    if hi <= lo:
        # Degenerate: every score identical in both windows.
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    # np.histogram right-inclusive on the last bin.
    b_counts, _ = np.histogram(baseline, bins=edges)
    l_counts, _ = np.histogram(live, bins=edges)
    base_share = np.maximum(b_counts.astype(np.float64), 0.5) / baseline.size
    live_share = np.maximum(l_counts.astype(np.float64), 0.5) / live.size
    return float(np.sum((live_share - base_share) * np.log(live_share / base_share)))


def _verdict(
    *,
    delta_certified: float,
    psi: float | None,
    ks_p: float | None,
) -> DriftVerdict:
    abs_delta = abs(delta_certified)
    alert = abs_delta >= _RATE_ALERT
    watch = abs_delta >= _RATE_WATCH
    if psi is not None:
        alert = alert or psi >= _PSI_ALERT
        watch = watch or psi >= _PSI_WATCH
    if ks_p is not None:
        alert = alert or ks_p < _KS_ALERT
        watch = watch or ks_p < _KS_WATCH
    if alert:
        return "recertify_suggested"
    if watch:
        return "watch"
    return "ok"


def run_drift_monitor(
    live: list[Decision],
    baseline: list[Decision],
    *,
    profile: CertificationProfile | None = None,
    n_bins: int = 10,
    min_window: int = 30,
) -> DriftReport:
    """
    Compare a live decisions window to a baseline window.

    Both windows should come from the same profile (same threshold / audit_rate)
    so the comparison is behavioural drift, not a config change. When `profile`
    is set, its `coverage_calibration` is attached as context and a note is added
    if the baseline certified rate disagrees with it substantially.
    """
    if len(live) < min_window:
        raise ValueError(
            f"live window has {len(live)} decisions; need at least {min_window}"
        )
    if len(baseline) < min_window:
        raise ValueError(
            f"baseline window has {len(baseline)} decisions; need at least {min_window}"
        )

    base_stats = window_stats(baseline)
    live_stats = window_stats(live)
    delta_c = live_stats.certified_rate - base_stats.certified_rate
    delta_a = live_stats.auto_rate - base_stats.auto_rate

    notes: list[str] = [
        "Label-free monitor: a shift is an alarm, not a certificate. "
        "Respond with a higher audit_rate and/or decisionops recertify.",
        "Certified rate counts reason in {certified, audit}, so audit sampling "
        "does not bias the comparison to profile coverage.",
    ]

    psi: float | None
    ks_stat: float | None
    ks_p: float | None
    try:
        psi = population_stability_index(
            _confidence_array(baseline), _confidence_array(live), n_bins=n_bins
        )
    except ValueError as exc:
        psi = None
        notes.append(f"PSI skipped: {exc}")
    ks = ks_2samp(_confidence_array(baseline), _confidence_array(live))
    ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)

    expected = None
    profile_hash = None
    if profile is not None:
        profile_hash = profile.profile_hash
        expected = profile.coverage_calibration
        if expected is not None and abs(base_stats.certified_rate - expected) > 0.08:
            notes.append(
                f"Baseline certified rate {base_stats.certified_rate:.1%} differs from "
                f"profile coverage_calibration {expected:.1%} by more than 8 points; "
                "confirm the baseline was produced with this profile."
            )
        live_hashes = {d.profile for d in live if d.profile}
        if live_hashes and profile_hash not in live_hashes:
            notes.append(
                f"Live decisions reference profile hash(es) {sorted(live_hashes)}; "
                f"monitor profile is {profile_hash}. Compare windows from one profile."
            )

    verdict = _verdict(delta_certified=delta_c, psi=psi, ks_p=ks_p)
    if verdict == "ok":
        notes.append(
            "No material shift in certified rate or confidence distribution. This does "
            "not mean the certificate still holds: concept drift (worse accuracy at the "
            "same confidence) is invisible here. Keep the scheduled recertify."
        )
    elif verdict == "watch":
        notes.append(
            "Moderate shift. Increase scrutiny or audit_rate; plan a recertify "
            "when audit labels mature."
        )
    else:
        notes.append(
            "Material shift. Prefer decisionops recertify on matured audit rows; "
            "consider a temporary rise in audit_rate until then."
        )
    if psi is not None:
        notes.append(
            f"Confidence PSI={psi:.3f} (watch≥{_PSI_WATCH}, alert≥{_PSI_ALERT})."
        )
    notes.append(
        f"KS p-value={ks_p:.4g} on confidence "
        f"(watch p<{_KS_WATCH}, alert p<{_KS_ALERT})."
    )
    notes.append(
        f"Δ certified rate={delta_c:+.1%} "
        f"(watch≥{_RATE_WATCH:.0%}, alert≥{_RATE_ALERT:.0%})."
    )

    return DriftReport(
        verdict=verdict,
        profile_hash=profile_hash,
        baseline=base_stats,
        live=live_stats,
        delta_certified_rate=delta_c,
        delta_auto_rate=delta_a,
        psi_confidence=psi,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        expected_certified_rate=expected,
        notes=notes,
        metadata={
            "n_bins": n_bins,
            "min_window": min_window,
            "psi_watch": _PSI_WATCH,
            "psi_alert": _PSI_ALERT,
            "rate_watch": _RATE_WATCH,
            "rate_alert": _RATE_ALERT,
            "ks_watch": _KS_WATCH,
            "ks_alert": _KS_ALERT,
        },
    )
