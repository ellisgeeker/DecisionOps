#!/usr/bin/env python3
"""
Measure the cascade label loop with closed-form true risk.

Certify a two-stage cascade on pre-drift traffic; drift; route drifted traffic
with the old cascade profile (audit share eps); recertify on audit labels only.
The cascade counterpart of `scripts/measure_label_loop.py` (findings §10).

Each stage k has confidence c_k ~ U(lo_k, 0.999), independent across stages,
and is right with probability c_k - delta, independently given c. Every choice
is automatic. True joint error of a point (l1, l2) is then closed form:

    R = E[1{c1 >= l1} (1 - c1 + delta)] + P(c1 < l1) E[1{c2 >= l2} (1 - c2 + delta)]

  pre:     delta=0     calibrated
  concept: delta=0.10  same scores, both stages lose 10 points at every c

  uv run python scripts/measure_cascade_label_loop.py             # 200 reps
  uv run python scripts/measure_cascade_label_loop.py --reps 20   # quick look
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decisionops.conformal.cascade_certify import run_cascade_certification
from decisionops.contract import DecisionContract
from decisionops.decide import cascade_profile_from_report, decide_cascade_many
from decisionops.recertify import run_cascade_recertification
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

HI = 0.999
LO = (0.55, 0.70)
CHOICES = ["a", "b", "c"]
CONTRACT = DecisionContract(name="cascade-loop", choices=CHOICES, auto_choices=CHOICES)


def _stage_term(lam: float | None, lo: float, delta: float) -> tuple[float, float]:
    """(E[1{c >= lam} * miss], P(c < lam)) for c ~ U(lo, HI), miss = 1 - c + delta."""
    width = HI - lo
    if lam is None:
        return 0.0, 1.0
    a = min(max(lam, lo), HI)
    # integral over [a, HI] of (1 + delta - c) dc / width
    err = ((1.0 + delta) * (HI - a) - (HI**2 - a**2) / 2.0) / width
    return err, (a - lo) / width


def true_risk(l1: float | None, l2: float | None, delta: float) -> float:
    e1, below1 = _stage_term(l1, LO[0], delta)
    e2, _ = _stage_term(l2, LO[1], delta)
    return e1 + below1 * e2


def _pool(n: int, rng: np.random.Generator, delta: float, prefix: str):
    cases, p1, p2 = [], [], []
    for i in range(n):
        cid = f"{prefix}{i}"
        cases.append(DecisionCase(id=cid, choices=CHOICES, label="a"))
        for lo, out, model in ((LO[0], p1, "m1"), (LO[1], p2, "m2")):
            c = float(rng.uniform(lo, HI))
            pred = "a" if rng.random() < c - delta else "b"
            rest = (1.0 - c) / 2.0
            probs = {k: rest for k in CHOICES}
            probs[pred] = c
            out.append(
                DecisionPrediction(id=cid, prediction=pred, probabilities=probs, model=model)
            )
    return cases, p1, p2


def one_rep(rep: int, *, alpha: float, delta: float, eps: float, n: int, level: float):
    rng = np.random.default_rng(10_000 + rep)
    cases, p1, p2 = _pool(n, rng, 0.0, "pre")
    report = run_cascade_certification(
        cases,
        p1,
        p2,
        CONTRACT,
        alpha=alpha,
        confidence_level=level,
        calibrate=False,
        cal_ratio=0.5,
        seed=rep,
    )
    if not report.certificate.feasible:
        return None
    live = cascade_profile_from_report(report, audit_rate=eps, audit_seed=rep)
    old = (live.stages[0].threshold, live.stages[1].threshold)

    cases_d, d1, d2 = _pool(n, rng, delta, "live")
    decisions = decide_cascade_many(d1, live, d2)
    outcomes = [OutcomeFeedback(id=c.id, label=c.label) for c in cases_d]
    rec = run_cascade_recertification(
        live, decisions=decisions, outcomes=outcomes, stage1=d1, stage2=d2
    )
    row = {
        "pre_old": true_risk(*old, 0.0),
        "drift_old": true_risk(*old, delta),
        "verdict": rec.verdict,
        "n_audit": rec.n_audit_calibration,
    }
    if rec.profile is not None:
        new = (rec.profile.stages[0].threshold, rec.profile.stages[1].threshold)
        row["drift_new"] = true_risk(*new, delta)
        row["coverage_new"] = rec.profile.coverage_calibration
    return row


def summarize(rows: list[dict], alpha: float) -> dict:
    def stat(key: str) -> dict | None:
        vals = np.array([r[key] for r in rows if key in r], dtype=float)
        if vals.size == 0:
            return None
        return {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "share_above_alpha": float((vals > alpha).mean()),
        }

    verdicts: dict[str, int] = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    cov = [r["coverage_new"] for r in rows if "coverage_new" in r]
    return {
        "pre_old": stat("pre_old"),
        "drift_old": stat("drift_old"),
        "drift_new": stat("drift_new"),
        "verdicts": verdicts,
        "mean_n_audit": float(np.mean([r["n_audit"] for r in rows])),
        "mean_coverage_new": float(np.mean(cov)) if cov else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--out", type=Path, default=Path("report/cascade-label-loop.json"))
    args = parser.parse_args()

    result: dict = {"config": vars(args) | {"out": str(args.out)}, "by_eps": {}}
    for eps in args.eps:
        rows = []
        n_infeasible_pre = 0
        for rep in range(args.reps):
            row = one_rep(
                rep, alpha=args.alpha, delta=args.delta, eps=eps, n=args.n, level=args.level
            )
            if row is None:
                n_infeasible_pre += 1
            else:
                rows.append(row)
        summary = summarize(rows, args.alpha)
        summary["n_pre_infeasible"] = n_infeasible_pre
        result["by_eps"][str(eps)] = summary
        print(f"eps={eps:g}: {json.dumps(summary)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
