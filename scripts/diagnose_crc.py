#!/usr/bin/env python3
"""
Decisive check on the CRC threshold rule.

The synthetic pool (confidence ~ U(0.55, 0.999), perfectly calibrated) has a
closed-form optimal threshold, so we can compare lambda_hat against lambda_star
directly instead of squinting at noisy error means.

    true_joint_error(lam) = ((1-lam)^2 - 0.001^2) / (2 * 0.449)
    lambda_star(alpha)    = 1 - sqrt(2 * (0.449*alpha + 5e-7))

lambda_hat < lambda_star  =>  anti-conservative (guarantee violated)

Also compares two acceptance rules. CRC's proof needs the loss right-continuous
in lambda; 1{conf >= lam} is left-continuous, 1{conf > lam} is right-continuous.

The CI check lives in `tests/test_crc_analytic.py`. It uses n_cal=150 (not 600)
so the B/(n+1) penalty is a visible share of alpha — otherwise a penalty-free
rule still passes. Keep both; they answer different questions.
"""

from __future__ import annotations

import math

import numpy as np

LO, HI = 0.55, 0.999
SPAN = HI - LO


def lambda_star(alpha: float) -> float:
    return 1.0 - math.sqrt(2.0 * (SPAN * alpha + (1 - HI) ** 2 / 2.0))


def true_joint_error(lam: float) -> float:
    return ((1.0 - lam) ** 2 - (1 - HI) ** 2) / (2.0 * SPAN)


def pool(rng, n):
    conf = rng.uniform(LO, HI, size=n)
    wrong = (rng.random(n) >= conf).astype(np.float64)
    return conf, wrong


def certify(conf, wrong, alpha, strict: bool, B=1.0):
    n = len(conf)
    if B / (n + 1) > alpha:
        return None
    for lam in np.unique(conf):
        acc = conf > lam if strict else conf >= lam
        rhat = float(wrong[acc].sum() / n)
        if (n / (n + 1.0)) * rhat + B / (n + 1.0) <= alpha:
            return float(lam)
    return None


def main(reps=300, n_cal=600):
    print(f"reps={reps}  n_cal={n_cal}\n")
    cols = (
        f"{'alpha':>6} {'rule':>5} {'lam*':>8} {'mean_lam':>9}"
        f" {'gap':>9} {'se':>8} {'z':>7} {'verdict':>11}"
    )
    print(cols)
    print("-" * len(cols))
    for alpha in (0.01, 0.05, 0.10):
        ls = lambda_star(alpha)
        for strict in (False, True):
            lams = []
            for rep in range(reps):
                c, w = pool(np.random.default_rng(90_000 + rep), n_cal)
                lam = certify(c, w, alpha, strict)
                if lam is not None:
                    lams.append(lam)
            a = np.asarray(lams)
            gap = float(a.mean() - ls)
            se = float(a.std(ddof=1) / math.sqrt(len(a)))
            z = gap / se if se > 0 else 0.0
            if z < -2.0:
                verdict = "ANTI-CONS"
            elif z > -1.0:
                verdict = "ok"
            else:
                verdict = "borderline"
            rule = ">" if strict else ">="
            implied = true_joint_error(float(a.mean()))
            print(
                f"{alpha:>6} {rule:>5} {ls:>8.5f} {a.mean():>9.5f}"
                f" {gap:>+9.5f} {se:>8.5f} {z:>+7.2f} {verdict:>11}"
                f"   implied_risk={implied:.5f}"
            )
    print("\nlam_hat should sit AT or ABOVE lam* (conservative).")
    print("z < -2 consistently means the rule is anti-conservative.")


if __name__ == "__main__":
    main()
