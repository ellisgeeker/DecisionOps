#!/usr/bin/env python3
"""
Can Jev's P(incident) separate cpython PRs that broke from those that did not?

Reports, on the enriched pack (incidents kept in full, ok PRs subsampled):

- AUC of P(incident) for incident vs ok;
- for a sweep of auto-merge thresholds on P(ok): the share of the population
  auto-merged and the population incident rate among / over everything, using
  each row's `metadata.weight` (inverse sampling rate) so the numbers are for
  the real 0.7% base rate, not the pack's 3%;
- the same split by change type (code / build-CI-tests / docs).

- `--crc-reps N`: CRC certificates on N population-rate resamples (all ok PRs
  plus as many incidents as a random draw of that size would hold), since the
  enriched pack itself is not exchangeable with live traffic;
- AUCs of trivial size baselines and a bootstrap 95% interval for Jev's.

  uv run python scripts/analyze_cpython_incidents.py \
    --dataset data/cpython/full/pack.jsonl --predictions data/cpython/full/jev.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CODE = ("Lib/", "Modules/", "Objects/", "Python/", "Include/", "Parser/", "PC/")


def change_type(state: str) -> str:
    files_line = next(line for line in state.splitlines() if line.startswith("Files changed"))
    files = files_line.split(": ", 1)[1].split(", ") if ": " in files_line else []
    if any(f.startswith(CODE) and not f.startswith("Lib/test/") for f in files):
        return "code"
    if any(f.startswith(("Doc/",)) for f in files) and not any(
        f.startswith((".github/", "Tools/", "Misc/", "Lib/test/", "PCbuild/")) for f in files
    ):
        return "docs"
    return "build_ci_tests"


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted half."""
    p, n = scores[pos], scores[~pos]
    if not len(p) or not len(n):
        return float("nan")
    order = np.argsort(np.concatenate([p, n]), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.concatenate([p, n])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(p)].sum()
    return float((r_pos - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--population-ok", type=int, default=None,
                        help="ok PRs in the population the pack was sampled from")
    parser.add_argument("--crc-reps", type=int, default=0)
    parser.add_argument("--alphas", default="0.003,0.005")
    args = parser.parse_args()

    cases = {c["id"]: c for c in map(json.loads, args.dataset.open())}
    preds = {p["id"]: p for p in map(json.loads, args.predictions.open())}
    ids = [i for i in cases if i in preds]
    p_inc = np.array([preds[i]["probabilities"]["incident"] for i in ids])
    y = np.array([cases[i]["label"] == "incident" for i in ids])
    w = np.array([cases[i]["metadata"]["weight"] for i in ids])
    kind = np.array([change_type(cases[i]["state"]) for i in ids])

    out: dict = {
        "n_scored": len(ids),
        "n_incident": int(y.sum()),
        "n_ok": int((~y).sum()),
        "population_incident_rate": float((w * y).sum() / w.sum()),
        "auc": auc(p_inc, y),
        "auc_by_type": {k: auc(p_inc[kind == k], y[kind == k]) for k in sorted(set(kind))},
        "n_by_type": {k: [int((y & (kind == k)).sum()), int((kind == k).sum())]
                      for k in sorted(set(kind))},
        "p_incident_quantiles": {
            "incident": np.round(np.quantile(p_inc[y], [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist(),
            "ok": np.round(np.quantile(p_inc[~y], [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist(),
        },
    }
    rows = []
    W = w.sum()
    for t in sorted(set(np.round(p_inc, 2)) | {1.01}):
        auto = p_inc < t  # auto-merge when P(incident) below the cut
        auto_share = float(w[auto].sum() / W)
        joint = float((w * (auto & y)).sum() / W)
        caught = float(1 - (auto & y).sum() / max(y.sum(), 1))
        rows.append(
            {
                "cut_p_incident": float(t),
                "auto_share": round(auto_share, 4),
                "joint_incident_rate": round(joint, 5),
                "incidents_escalated": round(caught, 3),
            }
        )
    out["sweep"] = rows

    import re

    rng = np.random.default_rng(0)

    def _boot(v: np.ndarray) -> list[float]:
        pi, ni = np.where(y)[0], np.where(~y)[0]
        bs = [
            auc(v[idx], y[idx])
            for idx in (
                np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
                for _ in range(1000)
            )
        ]
        return [round(float(np.quantile(bs, 0.025)), 3), round(float(np.quantile(bs, 0.975)), 3)]

    states = [cases[i]["state"] for i in ids]
    lines = np.log1p([sum(map(int, re.search(r"Lines: \+(\d+) -(\d+)", s).groups()))
                      for s in states])
    nfiles = np.log1p([int(re.search(r"Files changed \((\d+)\)", s)[1]) for s in states])
    out["baselines"] = {
        "jev": {"auc": out["auc"], "ci95": _boot(p_inc)},
        "log_lines": {"auc": auc(lines, y), "ci95": _boot(lines)},
        "log_files": {"auc": auc(nfiles, y), "ci95": _boot(nfiles)},
    }
    terciles = np.digitize(lines, np.quantile(lines, [1 / 3, 2 / 3]))
    out["jev_auc_by_size_tercile"] = [auc(p_inc[terciles == t], y[terciles == t]) for t in range(3)]

    if args.crc_reps:
        from decisionops.conformal.certify import run_certification
        from decisionops.contract import load_contract
        from decisionops.io import load_dataset, load_predictions

        if args.contract is None or args.population_ok is None:
            raise SystemExit("--crc-reps needs --contract and --population-ok")
        contract = load_contract(args.contract)
        dcases = load_dataset(args.dataset)
        dpreds = {p.id: p for p in load_predictions(args.predictions)}
        inc = [c for c in dcases if c.label == "incident"]
        ok = [c for c in dcases if c.label == "ok"]
        k = round(len(inc) * len(ok) / args.population_ok)
        crc = {"incidents_per_draw": k, "draw_rate": k / (k + len(ok)), "levels": {}}
        for alpha in (float(a) for a in args.alphas.split(",")):
            res = []
            for s in range(args.crc_reps):
                pick = [inc[i] for i in rng.choice(len(inc), k, replace=False)]
                d = pick + ok
                lv = run_certification(
                    d, [dpreds[c.id] for c in d], alphas=[alpha], contract=contract, seed=s
                ).levels[0]
                t = lv.test or {}
                res.append((lv.certificate.feasible, t.get("coverage"), t.get("joint_error_rate")))
            feas = [r for r in res if r[0] and r[1] is not None]
            cov = [r[1] for r in feas]
            err = [r[2] for r in feas]
            crc["levels"][str(alpha)] = {
                "feasible": f"{len(feas)}/{args.crc_reps}",
                "test_auto_mean": float(np.mean(cov)) if cov else None,
                "test_auto_min": float(min(cov)) if cov else None,
                "test_auto_max": float(max(cov)) if cov else None,
                "test_joint_incident_mean": float(np.mean(err)) if err else None,
            }
        out["crc_resample"] = crc
    text = json.dumps(out, indent=2, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=2, default=float))
    print("\ncut  auto%   joint-incident  incidents-escalated")
    for r in rows:
        print(f"{r['cut_p_incident']:.2f}  {r['auto_share']:6.1%}  {r['joint_incident_rate']:.4%}"
              f"       {r['incidents_escalated']:.0%}")


if __name__ == "__main__":
    main()
