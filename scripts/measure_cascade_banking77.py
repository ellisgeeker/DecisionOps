#!/usr/bin/env python3
"""
Measure certified two-stage cascade on BANKING77 (cnb → char_lr).

All intents are automatic (same demo contract shape as the shadow walkthrough).
Compares the cascade LTT certificate to each stage alone under the same
high-probability level, and to a single-model certificate at the cascade's
per-row level (δ/m).

  uv run python scripts/measure_cascade_banking77.py
  uv run python scripts/measure_cascade_banking77.py --reps 5 --limit 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decisionops.conformal.cascade_certify import run_cascade_certification
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract
from decisionops.io import load_dataset, load_predictions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "examples" / "banking77"


def _contract_from_cases(cases) -> DecisionContract:
    choices = list(cases[0].choices)
    return DecisionContract(
        name="banking77-cascade-demo",
        description="All intents automatic; cascade LTT on joint error. Demo only.",
        choices=choices,
        auto_choices=choices,
    )


def _subset(cases, *pred_lists, n: int | None, seed: int):
    if n is None or n >= len(cases):
        return cases, pred_lists
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    cases_out = [cases[i] for i in idx]
    preds_out = []
    for preds in pred_lists:
        by_id = {p.id: p for p in preds}
        preds_out.append([by_id[c.id] for c in cases_out])
        if len(preds_out[-1]) != len(cases_out):
            missing = len(cases_out) - len(preds_out[-1])
            raise SystemExit(f"prediction set missing {missing} ids after subset")
    return cases_out, tuple(preds_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence-level", type=float, default=0.9)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: pack/cascade-cert-measure.json).",
    )
    args = parser.parse_args()

    pack = args.pack
    ds = pack / "dataset.jsonl"
    cheap_path = pack / "providers" / "tfidf_cnb.jsonl"
    mid_path = pack / "providers" / "tfidf_char_lr.jsonl"
    for p in (ds, cheap_path, mid_path):
        if not p.exists():
            raise SystemExit(
                f"missing {p}\n"
                "  uv run python scripts/prepare_hf_intent_gates.py --pack banking77\n"
                "  uv run python scripts/export_multi_providers.py --pack banking77"
            )

    cases = load_dataset(ds)
    cheap = load_predictions(cheap_path)
    mid = load_predictions(mid_path)
    cases, (cheap, mid) = _subset(cases, cheap, mid, n=args.limit, seed=args.seed0)
    contract = _contract_from_cases(cases)

    cascade_rows = []
    cheap_alone = []
    mid_alone = []
    for rep in range(args.reps):
        seed = args.seed0 + rep
        report = run_cascade_certification(
            cases,
            cheap,
            mid,
            contract,
            stage_names=("cnb", "char_lr"),
            alpha=args.alpha,
            confidence_level=args.confidence_level,
            seed=seed,
            calibrate=True,
        )
        cert = report.certificate
        cascade_rows.append(
            {
                "seed": seed,
                "feasible": cert.feasible,
                "lambda1": cert.lambda1,
                "lambda2": cert.lambda2,
                "n_cal": cert.n_calibration,
                "n_rows": cert.n_rows,
                "per_row_level": cert.per_row_level,
                "cal_auto": cert.calibration.automation_rate if cert.calibration else None,
                "cal_joint": cert.calibration.joint_error_rate if cert.calibration else None,
                "cal_ucb": cert.calibration.upper_bound if cert.calibration else None,
                "test_auto": report.test.automation_rate if report.test else None,
                "test_joint": report.test.joint_error_rate if report.test else None,
                "test_n1": report.test.n_auto_stage1 if report.test else None,
                "test_n2": report.test.n_auto_stage2 if report.test else None,
                "held": report.bound_held_on_test,
                "T1": report.stages[0].temperature,
                "T2": report.stages[1].temperature,
            }
        )

        for name, preds, bucket in (
            ("cnb", cheap, cheap_alone),
            ("char_lr", mid, mid_alone),
        ):
            single = run_certification(
                cases,
                preds,
                alphas=[args.alpha],
                seed=seed,
                calibrate=True,
                contract=contract,
                confidence_level=args.confidence_level,
            )
            lv = single.levels[0]
            bucket.append(
                {
                    "seed": seed,
                    "name": name,
                    "feasible": lv.certificate.feasible,
                    "threshold": lv.certificate.threshold,
                    "cal_auto": lv.certificate.coverage_calibration,
                    "test_auto": lv.test["coverage"] if lv.test else None,
                    "test_joint": lv.test["joint_error_rate"] if lv.test else None,
                    "held": lv.bound_held_on_test,
                }
            )

    def _summ(rows, key_auto="test_auto", key_joint="test_joint"):
        feas = [r for r in rows if r["feasible"]]
        held = [r for r in feas if r.get("held") is True]
        autos = [r[key_auto] for r in feas if r.get(key_auto) is not None]
        joints = [r[key_joint] for r in feas if r.get(key_joint) is not None]
        return {
            "n_reps": len(rows),
            "n_feasible": len(feas),
            "feasible_rate": len(feas) / len(rows) if rows else 0.0,
            "held_rate_among_feasible": (len(held) / len(feas)) if feas else None,
            "mean_test_auto": float(np.mean(autos)) if autos else None,
            "mean_test_joint": float(np.mean(joints)) if joints else None,
        }

    both_stages = sum(
        1
        for r in cascade_rows
        if r["feasible"] and r["lambda1"] is not None and r["lambda2"] is not None
    )
    summary = {
        "pack": str(pack.relative_to(ROOT)) if pack.is_relative_to(ROOT) else str(pack),
        "n_cases": len(cases),
        "alpha": args.alpha,
        "confidence_level": args.confidence_level,
        "reps": args.reps,
        "stages": ["tfidf_cnb", "tfidf_char_lr"],
        "cascade": _summ(cascade_rows),
        "cascade_both_stages_used": both_stages,
        "cascade_mean_n_cal": float(np.mean([r["n_cal"] for r in cascade_rows])),
        "cascade_n_rows": cascade_rows[0]["n_rows"] if cascade_rows else None,
        "cascade_per_row_level": cascade_rows[0]["per_row_level"] if cascade_rows else None,
        "cnb_alone": _summ(cheap_alone),
        "char_lr_alone": _summ(mid_alone),
        "rows": {
            "cascade": cascade_rows,
            "cnb": cheap_alone,
            "char_lr": mid_alone,
        },
    }

    out = args.out or (pack / "cascade-cert-measure.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    c, a, b = summary["cascade"], summary["cnb_alone"], summary["char_lr_alone"]
    print(f"wrote {out}")
    print(
        f"cascade  feasible {c['n_feasible']}/{c['n_reps']}  "
        f"mean test auto={c['mean_test_auto']}  joint={c['mean_test_joint']}  "
        f"held={c['held_rate_among_feasible']}  both_stages={both_stages}"
    )
    print(
        f"cnb      feasible {a['n_feasible']}/{a['n_reps']}  "
        f"mean test auto={a['mean_test_auto']}  joint={a['mean_test_joint']}"
    )
    print(
        f"char_lr  feasible {b['n_feasible']}/{b['n_reps']}  "
        f"mean test auto={b['mean_test_auto']}  joint={b['mean_test_joint']}"
    )
    if cascade_rows:
        print(
            f"LTT rows m={summary['cascade_n_rows']}  "
            f"per-row level={summary['cascade_per_row_level']:.4f}  "
            f"mean n_cal={summary['cascade_mean_n_cal']:.0f}"
        )


if __name__ == "__main__":
    main()
