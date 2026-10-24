#!/usr/bin/env python3
"""
Paired A/B of the `certify` default: temperature calibration on vs off.

Each row runs the product path, `run_certification(calibrate=True)` against
`calibrate=False`, with the same seed and therefore the same test split. With
calibration on, 20% of the non-test pool is spent fitting T; with it off, all
of it goes to CRC. The metric is certified test coverage (0 when infeasible).
`diff` is off minus on; positive means turning calibration off helps. The band
is +/- 2 standard errors of the paired difference.

Multiclass rows reshuffle a fixed pool (they estimate *this pool*, see
findings §3 limits). Binary rows draw a fresh synthetic pool per seed.

Usage:
  uv run python scripts/ab_certify_calibration.py            # both
  uv run python scripts/ab_certify_calibration.py --only multi
"""

from __future__ import annotations

import argparse

import numpy as np

from decisionops.conformal import run_certification
from decisionops.io import load_dataset, load_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction

_B = "examples/banking77"
_C = "examples/clinc_oos"
MULTI = (
    ("b77 laya bundled T=0.10", f"{_B}/providers/laya_dataset.jsonl", f"{_B}/providers/laya.jsonl"),
    ("b77 laya no bundled T", f"{_B}/providers/laya_rawT_dataset.jsonl",
     f"{_B}/providers/laya_rawT.jsonl"),
    ("b77 jev (500)", f"{_B}/providers/jev_dataset.jsonl", f"{_B}/providers/jev.jsonl"),
    ("b77 tfidf_char_lr", f"{_B}/dataset.jsonl", f"{_B}/providers/tfidf_char_lr.jsonl"),
    ("b77 tfidf_svm_sigmoid", f"{_B}/dataset.jsonl", f"{_B}/providers/tfidf_svm_sigmoid.jsonl"),
    ("b77 tfidf_cnb", f"{_B}/dataset.jsonl", f"{_B}/providers/tfidf_cnb.jsonl"),
    ("clinc tfidf_char_lr", f"{_C}/dataset.jsonl", f"{_C}/providers/tfidf_char_lr.jsonl"),
    ("clinc tfidf_cnb", f"{_C}/dataset.jsonl", f"{_C}/providers/tfidf_cnb.jsonl"),
)
ALPHAS = (0.05, 0.10)


def _coverage(cases, preds, alpha: float, seed: int, calibrate: bool) -> float | None:
    level = run_certification(
        cases, preds, alphas=[alpha], seed=seed, calibrate=calibrate
    ).levels[0]
    return float(level.test["coverage"]) if level.certificate.feasible else None


def _summarise(name: str, alpha: float, pairs: list[tuple[float | None, float | None]]) -> None:
    on = np.array([p[0] or 0.0 for p in pairs])
    off = np.array([p[1] or 0.0 for p in pairs])
    diff = off - on
    band = 2.0 * diff.std(ddof=1) / np.sqrt(len(diff))
    k = len(pairs)
    f_on = sum(p[0] is not None for p in pairs)
    f_off = sum(p[1] is not None for p in pairs)
    print(
        f"{name:24s} a={alpha:<4} on={on.mean():.3f} ({f_on}/{k})  "
        f"off={off.mean():.3f} ({f_off}/{k})  diff={diff.mean():+.3f} ±{band:.3f}",
        flush=True,
    )


def run_multi(seeds: int) -> None:
    for name, ds, pr in MULTI:
        cases, preds = load_dataset(ds), load_predictions(pr)
        for alpha in ALPHAS:
            pairs = [
                (_coverage(cases, preds, alpha, s, True), _coverage(cases, preds, alpha, s, False))
                for s in range(seeds)
            ]
            _summarise(name, alpha, pairs)


def _binary_pool(rng: np.random.Generator, n: int, shape):
    cases, preds = [], []
    for i in range(n):
        pred = "skip" if rng.random() < 0.7 else "deep"
        other = "deep" if pred == "skip" else "skip"
        conf = float(rng.uniform(0.5, 0.999))
        label = pred if rng.random() < shape(conf) else other
        cases.append(DecisionCase(id=str(i), choices=["skip", "deep"], label=label))
        probs = {pred: conf, other: 1 - conf}
        preds.append(DecisionPrediction(id=str(i), prediction=pred, probabilities=probs))
    return cases, preds


def run_binary(seeds: int) -> None:
    for label, shape in (("overconfident", lambda c: c**2), ("calibrated", lambda c: c)):
        for n in (500, 2000):
            for alpha in ALPHAS:
                pairs = []
                for s in range(seeds):
                    cases, preds = _binary_pool(np.random.default_rng(1000 * n + s), n, shape)
                    pairs.append(
                        (_coverage(cases, preds, alpha, s, True),
                         _coverage(cases, preds, alpha, s, False))
                    )
                _summarise(f"binary {label} n={n}", alpha, pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=("multi", "binary"))
    parser.add_argument("--multi-seeds", type=int, default=20)
    parser.add_argument("--binary-seeds", type=int, default=30)
    args = parser.parse_args()
    if args.only != "binary":
        run_multi(args.multi_seeds)
    if args.only != "multi":
        run_binary(args.binary_seeds)


if __name__ == "__main__":
    main()
