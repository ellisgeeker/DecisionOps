#!/usr/bin/env python3
"""
Split-to-split CRC variation on a FIXED real-data pool.

This does **not** independently resample the data-generating process. The
BANKING77 held-out set is 3076 rows; every seed here reshuffles that same pool.
The average therefore estimates properties of *this pool*, not a distributional
expectation. That distinction is the methodology error recorded in
`docs/status.md` — do not describe these figures as "validated across seeds"
in the population sense.

What it *does* establish, cheaply:

- Whether Laya's infeasibility at 1% (and 10%) is a one-split fluke or a
  property of this pool.
- The range of certified coverage for models that *are* feasible, so a single
  seed's 75% is not silently treated as a point.

Usage:
  uv run python scripts/resample_crc.py
  uv run python scripts/resample_crc.py --seeds 40 --output report/crc-split-variation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from decisionops.adapters import StaticFileProvider
from decisionops.conformal.certify import run_certification
from decisionops.io import load_dataset

ROOT = Path(__file__).resolve().parents[1]

_B77 = "examples/banking77"
_DS = f"{_B77}/dataset.jsonl"
_PROV = f"{_B77}/providers"
MODELS = (
    ("word_lr", _DS, f"{_PROV}/tfidf_word_lr.jsonl"),
    ("char_lr", _DS, f"{_PROV}/tfidf_char_lr.jsonl"),
    ("svm_sig", _DS, f"{_PROV}/tfidf_svm_sigmoid.jsonl"),
    ("cnb", _DS, f"{_PROV}/tfidf_cnb.jsonl"),
    ("laya", f"{_PROV}/laya_dataset.jsonl", f"{_PROV}/laya.jsonl"),
)

ALPHAS = (0.01, 0.05, 0.10)


def _pct(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return {}
    def at(p: float) -> float:
        if n == 1:
            return s[0]
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[i]
    return {
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "p10": at(0.10),
        "p50": at(0.50),
        "p90": at(0.90),
        "max": s[-1],
    }


def run(seeds: int, seed0: int) -> dict:
    loaded: dict[str, tuple] = {}
    for name, ds, pred in MODELS:
        cases = load_dataset(ROOT / ds)
        preds = StaticFileProvider(str(ROOT / pred)).predict_many(cases)
        loaded[name] = (cases, preds)

    per_model: dict[str, dict] = {}
    t0 = time.perf_counter()
    for name, (cases, preds) in loaded.items():
        levels: dict[float, dict] = {
            a: {"feasible": 0, "held": 0, "coverage": [], "joint_error": [], "threshold": []}
            for a in ALPHAS
        }
        n_crc_cal = None
        temperature = []
        for i in range(seeds):
            report = run_certification(
                cases,
                preds,
                alphas=list(ALPHAS),
                seed=seed0 + i,
                calibrate=True,
            )
            n_crc_cal = report.split.n_calibration
            if report.temperature is not None:
                temperature.append(report.temperature)
            for level in report.levels:
                bucket = levels[level.certificate.alpha]
                if not level.certificate.feasible:
                    continue
                bucket["feasible"] += 1
                if level.bound_held_on_test:
                    bucket["held"] += 1
                assert level.test is not None
                bucket["coverage"].append(float(level.test["coverage"]))
                bucket["joint_error"].append(float(level.test["joint_error_rate"]))
                bucket["threshold"].append(float(level.certificate.threshold or 0.0))

        per_model[name] = {
            "n_crc_cal": n_crc_cal,
            "n_total": len(cases),
            "temperature": _pct(temperature) if temperature else None,
            "levels": {
                f"{a:.2f}": {
                    "n_feasible": levels[a]["feasible"],
                    "n_held": levels[a]["held"],
                    "feasible_rate": levels[a]["feasible"] / seeds,
                    "held_rate_among_feasible": (
                        levels[a]["held"] / levels[a]["feasible"]
                        if levels[a]["feasible"]
                        else None
                    ),
                    "coverage_test": _pct(levels[a]["coverage"]),
                    "joint_error_test": _pct(levels[a]["joint_error"]),
                    "threshold": _pct(levels[a]["threshold"]),
                }
                for a in ALPHAS
            },
        }
        feas = {f"{a:.0%}": per_model[name]["levels"][f"{a:.2f}"]["n_feasible"] for a in ALPHAS}
        print(f"{name:10} feasible {feas}  n_crc_cal={n_crc_cal}", flush=True)

    elapsed = time.perf_counter() - t0
    return {
        "protocol": "single_pool_split_variation",
        "disclaimer": (
            "Every seed reshuffles the same 3076-row BANKING77 held-out pool. "
            "Averages estimate this pool, not a distributional expectation."
        ),
        "n_seeds": seeds,
        "seed0": seed0,
        "alphas": list(ALPHAS),
        "elapsed_s": round(elapsed, 2),
        "models": per_model,
    }


def _print_tables(payload: dict) -> None:
    print()
    print(payload["disclaimer"])
    print(f"seeds={payload['n_seeds']}  seed0={payload['seed0']}  elapsed={payload['elapsed_s']}s")
    print()
    header = (
        f"{'model':<10} {'alpha':>5} {'feas':>6} {'held':>6} "
        f"{'cov mean':>9} {'cov p10':>8} {'cov p90':>8} "
        f"{'jerr mean':>10} {'jerr p90':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, block in payload["models"].items():
        for a in ALPHAS:
            lv = block["levels"][f"{a:.2f}"]
            cov = lv["coverage_test"]
            jerr = lv["joint_error_test"]
            if lv["n_feasible"] == 0:
                print(
                    f"{name:<10} {a:>5.0%} {lv['n_feasible']:>3}/{payload['n_seeds']:<2} "
                    f"{'—':>6} {'infeasible':>9}"
                )
                continue
            held = lv["n_held"]
            print(
                f"{name:<10} {a:>5.0%} {lv['n_feasible']:>3}/{payload['n_seeds']:<2} "
                f"{held:>3}/{lv['n_feasible']:<2} "
                f"{cov['mean']:>9.1%} {cov['p10']:>8.1%} {cov['p90']:>8.1%} "
                f"{jerr['mean']:>10.4f} {jerr['p90']:>9.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "crc-split-variation.json")
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be >= 1")

    payload = run(args.seeds, args.seed0)
    _print_tables(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
