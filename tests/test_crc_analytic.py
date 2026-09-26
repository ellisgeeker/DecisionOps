"""
CRC correctness against a closed-form optimum.

Promoted from `scripts/diagnose_crc.py`. On a perfectly calibrated pool with
confidence ~ U(0.55, 0.999) the optimal threshold is known analytically:

    true_joint_error(lam) = ((1-lam)^2 - (1-HI)^2) / (2 * SPAN)
    lambda_star(alpha)    = 1 - sqrt(2 * (SPAN*alpha + (1-HI)^2/2))

A correct rule puts lambda_hat AT or ABOVE lambda_star (conservative). Comparing
against that baseline is far sharper than checking whether empirical error means
land under alpha, which is the knife-edge trap documented in
`tests/test_conformal.py`.

**Why n_cal is small here.** The discriminating power of this test comes from the
B/(n+1) penalty being a visible share of alpha. At n_cal=600 the penalty is
~0.0017, and deleting it from the implementation entirely still leaves all three
levels passing a `gap > -2*se` check — a broken implementation goes unnoticed. At
n_cal=150 the penalty is ~0.0066 and the signal is unmissable. Measured, not
guessed: see the docstring on `test_penalty_term_is_load_bearing`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from decisionops.conformal import certify_threshold
from decisionops.schemas import DecisionCase, DecisionPrediction

LO, HI = 0.55, 0.999
SPAN = HI - LO

ALPHAS = (0.02, 0.05, 0.10)
N_CAL = 150
REPS = 60
SEED_BASE = 90_000


def lambda_star(alpha: float) -> float:
    """Threshold whose true joint error is exactly alpha."""
    return 1.0 - math.sqrt(2.0 * (SPAN * alpha + (1 - HI) ** 2 / 2.0))


def true_joint_error(lam: float) -> float:
    """Closed-form joint error of accepting everything at or above `lam`."""
    return ((1.0 - lam) ** 2 - (1 - HI) ** 2) / (2.0 * SPAN)


def _calibrated_pool(rng: np.random.Generator, n: int):
    """Binary pool where P(correct) == reported confidence, exactly."""
    pairs = []
    for i in range(n):
        conf = float(rng.uniform(LO, HI))
        correct = bool(rng.random() < conf)
        pairs.append(
            (
                DecisionCase(id=str(i), choices=["a", "b"], label="a" if correct else "b"),
                DecisionPrediction(
                    id=str(i),
                    prediction="a",
                    probabilities={"a": conf, "b": 1.0 - conf},
                ),
            )
        )
    return pairs


def _gap_stats(alpha: float, *, reps: int = REPS, n_cal: int = N_CAL, seed_base: int = SEED_BASE):
    """Mean lambda_hat - lambda_star over fresh pools, with its standard error."""
    lams = []
    for rep in range(reps):
        pairs = _calibrated_pool(np.random.default_rng(seed_base + rep), n_cal)
        result = certify_threshold(pairs, alpha)
        if result.feasible and result.threshold is not None:
            lams.append(result.threshold)

    assert len(lams) >= reps * 0.8, f"too many infeasible runs at alpha={alpha}"
    arr = np.asarray(lams)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean, mean - lambda_star(alpha), se


def test_lambda_hat_is_conservative_at_every_level():
    """
    lambda_hat sits above the analytic optimum at all three levels.

    The joint condition is what carries the test. No single level separates a
    correct implementation from a penalty-free one reliably — measured across six
    seed blocks, a broken rule still scored z = +0.80 at alpha=5% once. All three
    positive at once did not happen for the broken rule in any block.
    """
    gaps = {}
    for alpha in ALPHAS:
        mean, gap, se = _gap_stats(alpha)
        gaps[alpha] = (mean, gap, se)
        assert gap > 0.0, (
            f"alpha={alpha}: mean lambda_hat {mean:.5f} sits BELOW lambda*="
            f"{lambda_star(alpha):.5f} (gap {gap:+.5f}, se {se:.5f}) — "
            f"anti-conservative, guarantee violated"
        )

    # alpha=5% carries the largest margin; a weak-but-positive result there is
    # itself a signal worth failing on (z stayed above +2.9 in every seed block).
    _, gap, se = gaps[0.05]
    assert gap / se > 1.5, (
        f"alpha=5%: gap is positive but only {gap / se:.2f} se — expected a wide "
        f"margin here; suspect the B/(n+1) penalty"
    )


def test_implied_true_risk_stays_under_alpha():
    """The analytic risk at the mean lambda_hat must respect the budget."""
    for alpha in ALPHAS:
        mean, _, _ = _gap_stats(alpha)
        implied = true_joint_error(mean)
        assert implied <= alpha, (
            f"alpha={alpha}: implied true risk {implied:.5f} exceeds the budget "
            f"at mean lambda_hat={mean:.5f}"
        )


def test_penalty_term_is_load_bearing():
    """
    A penalty-free rule fails the joint check that the real one passes.

    This is the test that makes the two above meaningful: it pins down that the
    assertions actually have teeth. Deleting B/(n+1) leaves an ordinary empirical
    threshold, which is anti-conservative — but only detectably so when the
    penalty is a real share of alpha, which is why N_CAL is 150 and not 600.
    """
    n = N_CAL
    gaps = []
    for alpha in ALPHAS:
        lams = []
        for rep in range(REPS):
            pairs = _calibrated_pool(np.random.default_rng(SEED_BASE + rep), n)
            conf = np.array([p.confidence for _, p in pairs])
            wrong = np.array(
                [0.0 if p.prediction == c.label else 1.0 for c, p in pairs]
            )
            # Same sweep as certify_threshold, minus the B/(n+1) term.
            for lam in np.unique(conf):
                if float(wrong[conf >= lam].sum() / n) <= alpha:
                    lams.append(float(lam))
                    break
        arr = np.asarray(lams)
        gaps.append(float(arr.mean()) - lambda_star(alpha))

    assert not all(g > 0.0 for g in gaps), (
        f"penalty-free rule passed the conservativeness check (gaps {gaps}) — "
        f"the assertions in this module are not discriminating"
    )


def test_analytic_helpers_are_mutually_consistent():
    """lambda_star and true_joint_error must invert each other."""
    for alpha in ALPHAS:
        assert true_joint_error(lambda_star(alpha)) == pytest.approx(alpha)
