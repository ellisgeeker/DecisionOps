"""
Conformal Risk Control (CRC) for abstention thresholds.

Faithful to Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk Control*:
for a loss that is non-increasing in the parameter lambda and bounded by B,

    lambda_hat = inf { lambda : (n / (n + 1)) * Rhat_n(lambda) + B / (n + 1) <= alpha }

gives the finite-sample guarantee  E[L_{n+1}(lambda_hat)] <= alpha  over the draw of
both the calibration set and the new sample, assuming exchangeability.

**Which risk this controls.** The per-example loss used here is

    L_i(lambda) = w(y_i) * 1 { confidence_i >= lambda  and  prediction_i != y_i }

where `w(y) = severity.get(y, 1)` comes from the decision contract (default 1).
So the certified quantity is the **joint (marginal) weighted automated error**:
the average loss over *all* traffic that is auto-accepted *and* wrong, with
misses weighted by the true label's severity. It is NOT the conditional
selective risk (error rate *among* accepted samples) that
`decisionops.selective` reports empirically.

The two differ by coverage:

    selective_risk = joint_error_rate / coverage

Conditional selective risk is a ratio of two random quantities, which breaks the
monotone-bounded-loss assumption above, so CRC does not certify it directly. If
you need selective risk <= s and expect coverage around c, targeting a joint
error budget of roughly s * c is the usual move — but c is itself random, so that
conversion is a design heuristic, not a guarantee. This module reports the
empirical selective risk at lambda_hat for context and labels it as empirical.

**Expectation vs high probability.** CRC bounds E[loss]: averaged over draws of
the calibration set, risk is <= alpha, but any one certificate can sit above
it (roughly half do, since the smallest feasible lambda lands true risk just
under alpha). Passing `confidence_level=1-delta` switches to a Risk-Controlling
Prediction Set (Bates et al., 2021) with a Clopper-Pearson upper bound:

    lambda_hat = inf { lambda : UCB_delta(Rhat_n(lambda)) <= alpha }

    P( true joint error(lambda_hat) <= alpha ) >= 1 - delta

over the calibration draw. The scan runs from the strictest threshold down and
stops at the first one whose upper bound exceeds alpha (fixed-sequence
testing), so no multiplicity correction is needed. The bound is exact for 0-1
loss only, so every per-example loss mass must be 0 or 1 (`loss_bound` must be
1, and contract severities may not introduce fractional weights). It is
stricter than CRC: with zero errors, `n >= ln(delta) / ln(1 - alpha)` — 299
samples at alpha=1%, 95%.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import beta as beta_dist

from decisionops.schemas import DecisionCase, DecisionPrediction

GuaranteeKind = Literal["expectation", "high_probability"]
SeverityOf = Callable[[str], float]


class CRCResult(BaseModel):
    """Outcome of a CRC calibration run for one target risk level."""

    alpha: float
    """Target bound on the (weighted) joint automated error."""

    guarantee: GuaranteeKind = "expectation"
    """
    `expectation`: CRC, E[joint error] <= alpha over draws of cal + new sample.
    `high_probability`: RCPS / Clopper-Pearson, P(true joint error <= alpha)
    >= confidence_level over the draw of the calibration set.
    """

    confidence_level: float | None = None
    """1 - delta for `high_probability`; None for plain CRC."""

    controlled_quantity: str = "joint_automated_error_rate"
    threshold: float | None = None
    """Certified lambda_hat; None when no threshold can satisfy the bound."""

    feasible: bool
    n_calibration: int
    min_n_required: int
    """Smallest calibration size for which the bound is satisfiable at all."""

    loss_bound: float = 1.0
    empirical_joint_error: float | None = None
    """Rhat_n(lambda_hat) on the calibration set (not a guarantee)."""

    crc_bound_value: float | None = None
    """
    Bound actually checked against alpha at lambda_hat.

    CRC: (n/(n+1))*Rhat + B/(n+1). High-probability: the one-sided
    Clopper-Pearson upper limit on the true joint error at confidence_level.
    """

    coverage_calibration: float | None = None
    empirical_selective_risk_calibration: float | None = None
    """Mean loss among accepted calibration samples. Empirical, not certified."""

    n_accepted_calibration: int = 0
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def min_calibration_samples(alpha: float, loss_bound: float = 1.0) -> int:
    """
    Smallest n for which the CRC bound can be met even with zero empirical loss.

    The bound contains an irreducible B/(n+1) term, so alpha >= B/(n+1) is
    necessary: n >= B/alpha - 1. At alpha=1% with 0-1 loss that is 99 samples.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if loss_bound <= 0.0:
        raise ValueError(f"loss_bound must be > 0, got {loss_bound}")
    return max(1, math.ceil(loss_bound / alpha - 1.0))


def _check_confidence_level(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")


def min_calibration_samples_hp(alpha: float, confidence_level: float) -> int:
    """
    Smallest n for which the high-probability bound can hold with zero errors.

    With k=0 the Clopper-Pearson upper limit is 1 - delta^(1/n), so it drops to
    alpha once (1 - alpha)^n <= delta, i.e. n >= ln(delta) / ln(1 - alpha).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    _check_confidence_level(confidence_level)
    delta = 1.0 - confidence_level
    return max(1, math.ceil(math.log(delta) / math.log1p(-alpha)))


def clopper_pearson_upper(k: int, n: int, confidence_level: float) -> float:
    """One-sided upper confidence limit on a binomial rate: k events in n trials."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    _check_confidence_level(confidence_level)
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(confidence_level, k + 1, n - k))


def min_selective_accepted(alpha: float, confidence_level: float) -> int:
    """
    Fewest auto-accepted calibration cases a selective certificate can pass on.

    Same algebra as `min_calibration_samples_hp`, but over *accepted* cases:
    with zero errors among m accepted, the bound is <= alpha only once
    m >= ln(delta) / ln(1 - alpha) — 45 at alpha=5%, 90%; 59 at 95%.
    """
    return min_calibration_samples_hp(alpha, confidence_level)


def _controlled_quantity(severity_of: SeverityOf | None) -> str:
    if severity_of is None:
        return "joint_automated_error_rate"
    return "weighted_joint_automated_error"


def _confidence_and_wrong(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    auto_choices: frozenset[str] | None = None,
    severity_of: SeverityOf | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Confidence and per-case loss mass.

    With a contract, a case predicted as an escalation choice is never
    auto-accepted, so it carries zero loss at every lambda. Encoding that as
    confidence = -inf keeps it out of every acceptance set and out of the
    candidate grid, while `n` still counts it — coverage and joint error stay
    fractions of *all* traffic.

    A miss whose true label is `y` contributes `severity_of(y)` (default 1).
    """
    conf = np.array([pred.confidence for _, pred in pairs], dtype=np.float64)
    if severity_of is None:
        wrong = np.array(
            [0.0 if pred.prediction == case.label else 1.0 for case, pred in pairs],
            dtype=np.float64,
        )
    else:
        wrong = np.array(
            [
                0.0 if pred.prediction == case.label else float(severity_of(case.label))
                for case, pred in pairs
            ],
            dtype=np.float64,
        )
    if auto_choices is not None:
        eligible = np.array([pred.prediction in auto_choices for _, pred in pairs], dtype=bool)
        conf = np.where(eligible, conf, -np.inf)
    return conf, wrong


def _assert_zero_one_loss(wrong: np.ndarray, loss_bound: float) -> None:
    """Clopper-Pearson needs Bernoulli events; fractional severity is out."""
    if loss_bound != 1.0:
        raise ValueError(
            "confidence_level uses a binomial bound and requires 0-1 loss "
            f"(loss_bound=1), got {loss_bound}"
        )
    if not np.all((wrong == 0.0) | (wrong == 1.0)):
        raise ValueError(
            "confidence_level requires 0-1 loss; contract severity introduces "
            "weights other than 0 or 1"
        )


def _certify_high_probability(
    conf: np.ndarray,
    wrong: np.ndarray,
    alpha: float,
    confidence_level: float,
    auto_choices: frozenset[str] | None,
    *,
    controlled_quantity: str,
) -> CRCResult:
    """RCPS with a Clopper-Pearson bound; fixed-sequence scan from strict to loose."""
    n = int(conf.size)
    min_n = min_calibration_samples_hp(alpha, confidence_level)
    base = {
        "alpha": alpha,
        "guarantee": "high_probability",
        "confidence_level": confidence_level,
        "controlled_quantity": controlled_quantity,
        "n_calibration": n,
        "min_n_required": min_n,
        "loss_bound": 1.0,
    }
    notes: list[str] = []

    # With zero errors the bound is at its lowest; if that fails nothing can pass.
    if clopper_pearson_upper(0, n, confidence_level) > alpha:
        notes.append(
            f"Infeasible: even with zero errors the {confidence_level:.0%} upper bound on "
            f"n={n} samples exceeds alpha={alpha:.4f}. Need at least {min_n} "
            "calibration samples."
        )
        return CRCResult(threshold=None, feasible=False, notes=notes, **base)

    grid = np.unique(conf[np.isfinite(conf)])
    if grid.size == 0:
        notes.append("No calibration case is on the automatic path; nothing to certify.")
        return CRCResult(threshold=None, feasible=False, notes=notes, **base)

    chosen: float | None = None
    chosen_k = 0
    chosen_ucb = 0.0
    # Strictest first. Each test is at level delta; stopping at the first failure
    # keeps the family-wise error at delta because the loss is monotone.
    for lam in grid[::-1]:
        k = int(np.sum(wrong[conf >= lam]))
        ucb = clopper_pearson_upper(k, n, confidence_level)
        if ucb > alpha:
            break
        chosen, chosen_k, chosen_ucb = float(lam), k, ucb

    if chosen is None:
        notes.append(
            "Even the strictest observed threshold has an upper error bound above "
            "alpha; abstain on all traffic at this level."
        )
        return CRCResult(threshold=None, feasible=False, notes=notes, **base)

    accepted = conf >= chosen
    n_accepted = int(accepted.sum())
    coverage = n_accepted / n
    if coverage < 0.05:
        notes.append(
            f"Certified threshold covers only {coverage:.1%} of calibration traffic."
        )
    notes.append(
        f"High-probability certificate: with probability >= {confidence_level:.0%} over "
        "the calibration draw, the true joint automated error rate is <= alpha."
    )
    return CRCResult(
        threshold=chosen,
        feasible=True,
        empirical_joint_error=chosen_k / n,
        crc_bound_value=chosen_ucb,
        coverage_calibration=coverage,
        empirical_selective_risk_calibration=(
            float(wrong[accepted].mean()) if n_accepted else None
        ),
        n_accepted_calibration=n_accepted,
        notes=notes,
        metadata={
            "n_grid_candidates": int(grid.size),
            "auto_choices": sorted(auto_choices) if auto_choices is not None else None,
            "bound": "clopper_pearson",
        },
        **base,
    )


def certify_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    alpha: float,
    *,
    loss_bound: float = 1.0,
    auto_choices: frozenset[str] | None = None,
    severity_of: SeverityOf | None = None,
    confidence_level: float | None = None,
) -> CRCResult:
    """
    Choose the smallest threshold whose CRC bound holds at level `alpha`.

    `confidence_level=None` (default) is Conformal Risk Control: the joint error
    is bounded by alpha **in expectation**. A value in (0, 1), e.g. 0.95, gives a
    high-probability certificate instead: P(true joint error <= alpha) >=
    confidence_level over the calibration draw. That needs 0-1 loss.

    `pairs` must be the **calibration** split only. Using the same data to pick
    lambda_hat and to report performance voids the guarantee.

    Candidate thresholds are the distinct calibration confidences; the smallest
    feasible one is returned, which maximises coverage subject to the bound.

    `auto_choices` (from a DecisionContract) restricts acceptance to the cheap
    path. `severity_of(label)` weights a miss by its true label (default 1).
    The loss stays non-increasing in lambda and bounded by `loss_bound`, so the
    CRC guarantee is unchanged; escalations simply never count as loss.
    """
    if not pairs:
        raise ValueError("cannot certify on empty calibration set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if loss_bound <= 0.0:
        raise ValueError(f"loss_bound must be > 0, got {loss_bound}")

    conf, wrong = _confidence_and_wrong(pairs, auto_choices, severity_of)
    quantity = _controlled_quantity(severity_of)
    if confidence_level is not None:
        _check_confidence_level(confidence_level)
        _assert_zero_one_loss(wrong, loss_bound)
        return _certify_high_probability(
            conf,
            wrong,
            alpha,
            confidence_level,
            auto_choices,
            controlled_quantity=quantity,
        )

    n = len(pairs)
    min_n = min_calibration_samples(alpha, loss_bound)
    notes: list[str] = []

    # The B/(n+1) penalty alone can exceed alpha; then no threshold works.
    if loss_bound / (n + 1) > alpha:
        notes.append(
            f"Infeasible: the irreducible B/(n+1) penalty is "
            f"{loss_bound / (n + 1):.4f} > alpha={alpha:.4f}. "
            f"Need at least {min_n} calibration samples."
        )
        return CRCResult(
            alpha=alpha,
            controlled_quantity=quantity,
            threshold=None,
            feasible=False,
            n_calibration=n,
            min_n_required=min_n,
            loss_bound=loss_bound,
            notes=notes,
        )

    # Rhat is a step function of lambda; steps occur at observed confidences.
    # Escalated cases (-inf) are never candidates.
    grid = np.unique(conf[np.isfinite(conf)])
    if grid.size == 0:
        notes.append("No calibration case is on the automatic path; nothing to certify.")
        return CRCResult(
            alpha=alpha,
            controlled_quantity=quantity,
            threshold=None,
            feasible=False,
            n_calibration=n,
            min_n_required=min_n,
            loss_bound=loss_bound,
            notes=notes,
        )
    chosen: float | None = None
    chosen_rhat = 0.0
    chosen_bound = 0.0

    for lam in grid:
        accepted = conf >= lam
        rhat = float(np.sum(wrong[accepted]) / n)
        bound = (n / (n + 1.0)) * rhat + loss_bound / (n + 1.0)
        if bound <= alpha:
            chosen = float(lam)
            chosen_rhat = rhat
            chosen_bound = bound
            break

    if chosen is None:
        # Even the strictest observed threshold keeps too much error mass.
        notes.append(
            "No calibration confidence level satisfies the bound; abstain on all "
            "traffic or improve the model before automating at this alpha."
        )
        return CRCResult(
            alpha=alpha,
            controlled_quantity=quantity,
            threshold=None,
            feasible=False,
            n_calibration=n,
            min_n_required=min_n,
            loss_bound=loss_bound,
            notes=notes,
        )

    accepted = conf >= chosen
    n_accepted = int(accepted.sum())
    coverage = n_accepted / n
    sel_risk = float(wrong[accepted].mean()) if n_accepted else None

    if n_accepted == 0:
        notes.append("Certified threshold accepts nothing on calibration data.")
    elif coverage < 0.05:
        notes.append(
            f"Certified threshold covers only {coverage:.1%} of calibration traffic."
        )
    if severity_of is None:
        notes.append(
            "Certified quantity is the joint automated error rate "
            "(auto-accepted AND wrong, over all traffic), not conditional selective risk."
        )
    else:
        notes.append(
            "Certified quantity is the severity-weighted joint automated error "
            "(auto-accepted AND wrong, weighted by true-label severity), not "
            "conditional selective risk."
        )

    return CRCResult(
        alpha=alpha,
        controlled_quantity=quantity,
        threshold=chosen,
        feasible=True,
        n_calibration=n,
        min_n_required=min_n,
        loss_bound=loss_bound,
        empirical_joint_error=chosen_rhat,
        crc_bound_value=chosen_bound,
        coverage_calibration=coverage,
        empirical_selective_risk_calibration=sel_risk,
        n_accepted_calibration=n_accepted,
        notes=notes,
        metadata={
            "n_grid_candidates": int(grid.size),
            "auto_choices": sorted(auto_choices) if auto_choices is not None else None,
        },
    )


def evaluate_certified_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    threshold: float,
    *,
    auto_choices: frozenset[str] | None = None,
    severity_of: SeverityOf | None = None,
) -> dict[str, float | int]:
    """
    Measure a certified threshold on held-out data.

    Reports the joint error (the certified quantity, severity-weighted when a
    contract supplies weights) alongside coverage and empirical selective risk,
    so the guarantee can be checked against reality.
    """
    if not pairs:
        raise ValueError("cannot evaluate on empty set")
    conf, wrong = _confidence_and_wrong(pairs, auto_choices, severity_of)
    n = len(pairs)
    accepted = conf >= threshold
    n_accepted = int(accepted.sum())
    joint_error = float(np.sum(wrong[accepted]) / n)
    return {
        "threshold": float(threshold),
        "n_total": n,
        "n_accepted": n_accepted,
        "coverage": n_accepted / n,
        "joint_error_rate": joint_error,
        "selective_risk": float(wrong[accepted].mean()) if n_accepted else float("nan"),
    }


SELECTIVE_QUANTITY = "selective_automated_error_rate"


def certify_selective_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    alpha: float,
    *,
    confidence_level: float = 0.9,
    auto_choices: frozenset[str] | None = None,
) -> CRCResult:
    """
    Certify the error rate **among auto-accepted cases**: Learn-Then-Test.

    This is the question customers ask ("of what you automate, is the error
    rate <= alpha?"), unlike `certify_threshold`, which bounds the share of
    *all* traffic that is auto-accepted and wrong. Selective risk is a ratio of
    two random counts, so CRC's monotone-loss argument does not apply. Instead,
    for each candidate lambda test

        H0(lambda): P(wrong | accepted at lambda) > alpha

    with a one-sided Clopper-Pearson bound on k errors among the m accepted
    calibration cases (exact, given m), and reject when that bound is <= alpha.

    **Fixed-sequence testing, strict to loose.** Candidates are the distinct
    calibration confidences, scanned from the highest down; the scan stops at
    the first lambda whose bound exceeds alpha. Each test runs at level delta,
    and the family-wise error is still <= delta because the order is fixed
    before looking at any label. The scan **starts** at the strictest lambda
    that accepts at least `min_selective_accepted(alpha, level)` cases: with
    fewer, even zero errors cannot pass, so stopping there would certify
    nothing. That start depends only on confidences, never on labels, so the
    order stays pre-specified and the guarantee holds:

        P( selective risk(lambda_hat) <= alpha ) >= confidence_level

    over the calibration draw. High-probability only: there is no
    in-expectation version, and 0-1 loss only (contract severity is ignored).
    """
    if not pairs:
        raise ValueError("cannot certify on empty calibration set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    _check_confidence_level(confidence_level)

    conf, wrong = _confidence_and_wrong(pairs, auto_choices)
    n = int(conf.size)
    min_m = min_selective_accepted(alpha, confidence_level)
    base = {
        "alpha": alpha,
        "guarantee": "high_probability",
        "confidence_level": confidence_level,
        "controlled_quantity": SELECTIVE_QUANTITY,
        "n_calibration": n,
        "min_n_required": min_m,
        "loss_bound": 1.0,
    }
    notes: list[str] = []

    grid = np.unique(conf[np.isfinite(conf)])[::-1]  # strict -> loose
    if grid.size == 0:
        notes.append("No calibration case is on the automatic path; nothing to certify.")
        return CRCResult(threshold=None, feasible=False, notes=notes, **base)

    accepted_counts = np.array([int(np.sum(conf >= lam)) for lam in grid])
    # First (strictest) lambda with enough accepted cases to be passable at all.
    passable = accepted_counts >= min_m
    start = int(np.argmax(passable)) if passable.any() else None
    if start is None:
        notes.append(
            f"Infeasible: only {int(accepted_counts.max())} calibration cases are on the "
            f"automatic path; a selective certificate at alpha={alpha:.4f} and "
            f"{confidence_level:.0%} needs at least {min_m} accepted cases even with zero "
            "errors."
        )
        return CRCResult(threshold=None, feasible=False, notes=notes, **base)

    chosen: float | None = None
    chosen_k = chosen_m = 0
    chosen_ucb = 0.0
    for lam, m in zip(grid[start:], accepted_counts[start:], strict=True):
        k = int(np.sum(wrong[conf >= lam]))
        ucb = clopper_pearson_upper(k, int(m), confidence_level)
        if ucb > alpha:
            break
        chosen, chosen_k, chosen_m, chosen_ucb = float(lam), k, int(m), ucb

    if chosen is None:
        notes.append(
            f"The strictest threshold with >= {min_m} accepted cases already has an "
            f"upper selective-error bound above alpha={alpha:.4f}; abstain at this level."
        )
        return CRCResult(threshold=None, feasible=False, notes=notes, **base)

    coverage = chosen_m / n
    if coverage < 0.05:
        notes.append(f"Certified threshold covers only {coverage:.1%} of calibration traffic.")
    notes.append(
        f"Selective certificate: with probability >= {confidence_level:.0%} over the "
        "calibration draw, the error rate among auto-accepted cases is <= alpha."
    )
    return CRCResult(
        threshold=chosen,
        feasible=True,
        empirical_joint_error=chosen_k / n,
        crc_bound_value=chosen_ucb,
        coverage_calibration=coverage,
        empirical_selective_risk_calibration=chosen_k / chosen_m,
        n_accepted_calibration=chosen_m,
        notes=notes,
        metadata={
            "n_grid_candidates": int(grid.size),
            "scan_start_index": start,
            "auto_choices": sorted(auto_choices) if auto_choices is not None else None,
            "bound": "clopper_pearson_selective",
            "method": "learn_then_test_fixed_sequence",
        },
        **base,
    )
