"""
Conformal Risk Control (CRC) for abstention thresholds.

Faithful to Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk Control*:
for a loss that is non-increasing in the parameter lambda and bounded by B,

    lambda_hat = inf { lambda : (n / (n + 1)) * Rhat_n(lambda) + B / (n + 1) <= alpha }

gives the finite-sample guarantee  E[L_{n+1}(lambda_hat)] <= alpha  over the draw of
both the calibration set and the new sample, assuming exchangeability.

**Which risk this controls.** The per-example loss used here is

    L_i(lambda) = 1 { confidence_i >= lambda  and  prediction_i != label_i }

so the certified quantity is the **joint (marginal) automated error rate**: the
fraction of *all* traffic that gets auto-accepted *and* is wrong. It is NOT the
conditional selective risk (error rate *among* accepted samples) that
`decisionops.selective` reports empirically.

The two differ by coverage:

    selective_risk = joint_error_rate / coverage

Conditional selective risk is a ratio of two random quantities, which breaks the
monotone-bounded-loss assumption above, so CRC does not certify it directly. If
you need selective risk <= s and expect coverage around c, targeting a joint
error budget of roughly s * c is the usual move — but c is itself random, so that
conversion is a design heuristic, not a guarantee. This module reports the
empirical selective risk at lambda_hat for context and labels it as empirical.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from decisionops.schemas import DecisionCase, DecisionPrediction


class CRCResult(BaseModel):
    """Outcome of a CRC calibration run for one target risk level."""

    alpha: float
    """Target bound on the joint automated error rate."""

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
    """(n/(n+1))*Rhat + B/(n+1) at lambda_hat; must be <= alpha."""

    coverage_calibration: float | None = None
    empirical_selective_risk_calibration: float | None = None
    """Error rate among accepted calibration samples. Empirical, not certified."""

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


def _confidence_and_wrong(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([pred.confidence for _, pred in pairs], dtype=np.float64)
    wrong = np.array(
        [0.0 if pred.prediction == case.label else 1.0 for case, pred in pairs],
        dtype=np.float64,
    )
    return conf, wrong


def certify_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    alpha: float,
    *,
    loss_bound: float = 1.0,
) -> CRCResult:
    """
    Choose the smallest threshold whose CRC bound holds at level `alpha`.

    `pairs` must be the **calibration** split only. Using the same data to pick
    lambda_hat and to report performance voids the guarantee.

    Candidate thresholds are the distinct calibration confidences; the smallest
    feasible one is returned, which maximises coverage subject to the bound.
    """
    if not pairs:
        raise ValueError("cannot certify on empty calibration set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    conf, wrong = _confidence_and_wrong(pairs)
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
            threshold=None,
            feasible=False,
            n_calibration=n,
            min_n_required=min_n,
            loss_bound=loss_bound,
            notes=notes,
        )

    # Rhat is a step function of lambda; steps occur at observed confidences.
    grid = np.unique(conf)
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
    notes.append(
        "Certified quantity is the joint automated error rate "
        "(auto-accepted AND wrong, over all traffic), not conditional selective risk."
    )

    return CRCResult(
        alpha=alpha,
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
        metadata={"n_grid_candidates": int(grid.size)},
    )


def evaluate_certified_threshold(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    threshold: float,
) -> dict[str, float | int]:
    """
    Measure a certified threshold on held-out data.

    Reports the joint error rate (the certified quantity) alongside coverage and
    empirical selective risk, so the guarantee can be checked against reality.
    """
    if not pairs:
        raise ValueError("cannot evaluate on empty set")
    conf, wrong = _confidence_and_wrong(pairs)
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
