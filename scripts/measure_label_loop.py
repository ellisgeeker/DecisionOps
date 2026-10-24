#!/usr/bin/env python3
"""
Measure the label loop end to end with closed-form true risk.

Certify on pre-drift traffic; drift; route the drifted traffic with the old
profile (audit share eps); recertify on audit labels only. The true joint
error of the old and new thresholds is computed analytically, so no test-split
noise enters the comparison.

Pools: confidence c ~ U(lo, 0.999), P(correct | c) = c - delta, and every
prediction is on the automatic path.

  pre:     lo=0.55, delta=0     calibrated
  concept: lo=0.55, delta=0.10  same scores, accuracy drops at every c
  score:   lo=0.75, delta=0     still calibrated, traffic mass moves up

Contrast: "naive" recertify on organic escalations only (the biased sample).

  uv run python scripts/measure_label_loop.py             # 200 reps, a few minutes
  uv run python scripts/measure_label_loop.py --reps 20   # quick look
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decisionops.conformal.certify import run_certification
from decisionops.conformal.risk_control import certify_threshold
from decisionops.contract import DecisionContract
from decisionops.decide import decide_many, profile_from_report
from decisionops.drift import run_drift_monitor
from decisionops.recertify import run_recertification
from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback

HI = 0.999
ROUTING = DecisionContract(name="routing", choices=["skip", "deep"], auto_choices=["skip"])
SCENARIOS = {"pre": (0.55, 0.0), "concept": (0.55, 0.10), "score": (0.75, 0.0)}


def true_risk(lam: float | None, lo: float, delta: float) -> float:
    """E[1{c >= lam} * (1 - (c - delta))] for c ~ U(lo, HI); None = abstain."""
    if lam is None:
        return 0.0
    m = min(max(lam, lo), HI)
    return ((1 + delta) * (HI - m) - (HI * HI - m * m) / 2) / (HI - lo)


def coverage(lam: float | None, lo: float) -> float:
    if lam is None:
        return 0.0
    return (HI - min(max(lam, lo), HI)) / (HI - lo)


def pool(rng: np.random.Generator, n: int, lo: float, delta: float, prefix: str):
    conf = rng.uniform(lo, HI, n)
    ok = rng.random(n) < conf - delta
    cases, preds = [], []
    for i in range(n):
        cid, c = f"{prefix}-{i}", float(conf[i])
        label = "skip" if ok[i] else "deep"
        cases.append(DecisionCase(id=cid, choices=["skip", "deep"], label=label))
        preds.append(
            DecisionPrediction(id=cid, prediction="skip", probabilities={"skip": c, "deep": 1 - c})
        )
    return cases, preds


def one_rep(rep: int, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed + rep)
    pre_cases, pre_preds = pool(rng, args.n_pre, *SCENARIOS["pre"], f"r{rep}-pre")
    report = run_certification(
        pre_cases, pre_preds, alphas=[args.alpha], contract=ROUTING, calibrate=False, seed=rep
    )
    profile = profile_from_report(report, args.alpha)
    out: dict = {"pre_old": true_risk(profile.threshold, *SCENARIOS["pre"])}
    baseline = decide_many(pre_preds, profile)
    for name in ("concept", "score"):
        lo, delta = SCENARIOS[name]
        cases, preds = pool(rng, args.n_post, lo, delta, f"r{rep}-{name}")
        label = {c.id: c.label for c in cases}
        plain = decide_many(preds, profile)
        row: dict = {
            "old": true_risk(profile.threshold, lo, delta),
            "old_cov": coverage(profile.threshold, lo),
            "drift": run_drift_monitor(plain, baseline, profile=profile).verdict,
        }
        # Naive: label every organic escalation. Many more rows, all below λ_old.
        organic = [
            (c, p) for c, p, d in zip(cases, preds, plain, strict=True) if d.action == "escalate"
        ]
        # Empty when λ_old escalates nothing; that rep has no naive certificate.
        naive = (
            certify_threshold(organic, args.alpha, auto_choices=ROUTING.auto_set)
            if organic
            else None
        )
        feasible = naive is not None and naive.feasible
        row["naive"] = true_risk(naive.threshold, lo, delta) if feasible else None
        row["naive_n"] = len(organic)
        row["naive_below_old"] = not feasible or naive.threshold < profile.threshold
        for eps in args.eps:
            prof = profile.model_copy(update={"audit_rate": eps, "audit_seed": rep})
            decisions = decide_many(preds, prof)
            outcomes = [OutcomeFeedback(id=d.id, label=label[d.id]) for d in decisions if d.audit]
            for mode, level in (("crc", None), ("hp", args.hp_level)):
                rec = run_recertification(
                    prof, decisions=decisions, outcomes=outcomes, predictions=preds,
                    contract=ROUTING, confidence_level=level,
                )
                lam = rec.profile.threshold if rec.profile is not None else None
                row[f"{mode}_{eps:g}"] = {
                    "n": rec.n_audit_calibration,
                    "feasible": lam is not None,
                    "risk": true_risk(lam, lo, delta),
                    "cov": coverage(lam, lo),
                }
        out[name] = row
    return out


def _stats(values, alpha: float) -> dict | None:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p90": float(np.quantile(v, 0.9)),
        "above_alpha": float((v > alpha).mean()),
    }


def summarise(rows: list[dict], args: argparse.Namespace) -> dict:
    a = args.alpha
    summary: dict = {"pre_old": _stats((r["pre_old"] for r in rows), a)}
    for name in ("concept", "score"):
        rs = [r[name] for r in rows]
        sc: dict = {
            "old": _stats((x["old"] for x in rs), a),
            "old_cov": float(np.mean([x["old_cov"] for x in rs])),
            "naive": _stats((x["naive"] for x in rs), a),
            "naive_n": float(np.mean([x["naive_n"] for x in rs])),
            "naive_always_below_old": all(x["naive_below_old"] for x in rs),
            "drift": {
                v: sum(x["drift"] == v for x in rs) for v in ("ok", "watch", "recertify_suggested")
            },
        }
        for eps in args.eps:
            for mode in ("crc", "hp"):
                cells = [x[f"{mode}_{eps:g}"] for x in rs]
                feas = [c for c in cells if c["feasible"]]
                sc[f"{mode}_{eps:g}"] = {
                    "n_audit": float(np.mean([c["n"] for c in cells])),
                    "feasible": len(feas),
                    "risk": _stats((c["risk"] for c in feas), a),
                    "cov": float(np.mean([c["cov"] for c in feas])) if feas else None,
                }
        summary[name] = sc
    return summary


def _fmt(s: dict | None) -> str:
    if s is None:
        return "—"
    return f"mean {s['mean']:.4f}  p90 {s['p90']:.4f}  >α {s['above_alpha']:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--n-pre", type=int, default=3000)
    parser.add_argument("--n-post", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eps", default="0.05,0.1,0.2", help="Audit rates, comma-separated")
    parser.add_argument("--hp-level", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=60_000)
    parser.add_argument("--out", type=Path, default=Path("report/label-loop.json"))
    args = parser.parse_args()
    args.eps = [float(x) for x in args.eps.split(",")]

    rows = []
    for rep in range(args.reps):
        rows.append(one_rep(rep, args))
        if (rep + 1) % 20 == 0:
            print(f"  {rep + 1}/{args.reps} reps", flush=True)
    summary = summarise(rows, args)

    print(f"\nα={args.alpha:g}  reps={args.reps}  n_pre={args.n_pre}  n_post={args.n_post}")
    print(f"pre-drift, old λ:  {_fmt(summary['pre_old'])}")
    for name in ("concept", "score"):
        sc = summary[name]
        print(f"\n[{name}]  drift monitor: {sc['drift']}")
        print(f"  old λ (no recertify)    {_fmt(sc['old'])}  cov {sc['old_cov']:.1%}")
        print(f"  naive organic (n≈{sc['naive_n']:.0f})  {_fmt(sc['naive'])}  "
              f"λ below old in every rep: {sc['naive_always_below_old']}")
        for eps in args.eps:
            for mode in ("crc", "hp"):
                c = sc[f"{mode}_{eps:g}"]
                cov = "—" if c["cov"] is None else f"{c['cov']:.1%}"
                print(f"  recertify {mode} ε={eps:g} (n≈{c['n_audit']:.0f}, "
                      f"feasible {c['feasible']}/{args.reps})  {_fmt(c['risk'])}  cov {cov}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: v for k, v in vars(args).items() if k != "out"}, "summary": summary}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
