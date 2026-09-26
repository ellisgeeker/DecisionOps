#!/usr/bin/env python3
"""
Export predictions from several sklearn heads for an existing DecisionOps pack,
so `decisionops compare` can score them under one shared split.

Reuses the pack's dataset.jsonl (labels + choices) and refits heads on the
original HF train split, so no eval leakage is introduced.

Usage:
  uv sync --extra bench
  uv run python scripts/export_multi_providers.py --pack banking77
  uv run python scripts/export_multi_providers.py --pack clinc --run-compare
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `datasets`. Install with: uv sync --extra bench"
        ) from exc
    return load_dataset


def _load_pack_cases(pack_dir: Path) -> list[dict]:
    path = pack_dir / "dataset.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build the pack first:\n"
            f"  uv run python scripts/prepare_hf_intent_gates.py --pack <name>"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hf_train_banking77() -> tuple[list[str], list[str]]:
    load_dataset = _require_datasets()
    ds = load_dataset("mteb/banking77")
    texts, labels = [], []
    for row in ds["train"]:
        texts.append(str(row["text"]).strip())
        labels.append(str(row["label_text"]))
    return texts, labels


def _hf_train_clinc(config: str = "plus") -> tuple[list[str], list[str]]:
    load_dataset = _require_datasets()
    ds = load_dataset("clinc/clinc_oos", config)
    names = ds["train"].features["intent"].names
    texts, labels = [], []
    for row in ds["train"]:
        texts.append(str(row["text"]).strip())
        lab = str(names[int(row["intent"])])
        labels.append("oos" if lab.lower() == "oos" else lab)
    return texts, labels


def _heads(seed: int) -> dict[str, Pipeline]:
    """Several 'providers' with genuinely different probability behaviour."""
    return {
        "tfidf_word_lr": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
            ]
        ),
        "tfidf_char_lr": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                    ),
                ),
                ("clf", LogisticRegression(max_iter=3000, C=2.0, random_state=seed)),
            ]
        ),
        "tfidf_svm_sigmoid": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
                    ),
                ),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(C=0.5, random_state=seed),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "tfidf_cnb": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)),
                ("clf", ComplementNB(alpha=0.3)),
            ]
        ),
    }


def _write_predictions(
    path: Path,
    ids: list[str],
    choices: list[str],
    classes: list[str],
    proba: np.ndarray,
    provider: str,
    model_name: str,
    latency_ms: float | None,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_correct_probs = 0
    with path.open("w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            probs = {str(c): float(proba[i, j]) for j, c in enumerate(classes)}
            for ch in choices:
                probs.setdefault(ch, 0.0)
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            pred = max(probs, key=probs.get)
            n_correct_probs += 1
            f.write(
                json.dumps(
                    {
                        "id": cid,
                        "prediction": pred,
                        "probabilities": probs,
                        "latency_ms": latency_ms,
                        "provider": provider,
                        "model": model_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {"rows": n_correct_probs, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=["banking77", "clinc"], required=True)
    parser.add_argument("--clinc-config", default="plus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-root", type=Path, default=ROOT / "examples")
    parser.add_argument("--run-compare", action="store_true")
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    args = parser.parse_args()

    pack_dir = args.examples_root / ("banking77" if args.pack == "banking77" else "clinc_oos")
    cases = _load_pack_cases(pack_dir)
    ids = [c["id"] for c in cases]
    texts_eval = [c["state"] for c in cases]
    choices = cases[0]["choices"]

    if args.pack == "banking77":
        texts_train, labels_train = _hf_train_banking77()
    else:
        texts_train, labels_train = _hf_train_clinc(args.clinc_config)

    out_dir = pack_dir / "providers"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, pipe in _heads(args.seed).items():
        print(f"\n--- fitting {name} on {len(texts_train)} train rows ---")
        t0 = time.perf_counter()
        pipe.fit(texts_train, labels_train)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        proba = pipe.predict_proba(texts_eval)
        infer_s = time.perf_counter() - t1
        per_row_ms = (infer_s / max(len(texts_eval), 1)) * 1000.0

        clf = pipe.named_steps["clf"]
        classes = [str(c) for c in clf.classes_]
        path = out_dir / f"{name}.jsonl"
        info = _write_predictions(
            path,
            ids=ids,
            choices=choices,
            classes=classes,
            proba=proba,
            provider="sklearn",
            model_name=name,
            latency_ms=round(per_row_ms, 3),
        )
        acc = float(
            np.mean(
                [
                    classes[int(np.argmax(proba[i]))] == cases[i]["label"]
                    for i in range(len(cases))
                ]
            )
        )
        rec = {
            "name": name,
            "fit_seconds": round(fit_s, 2),
            "infer_ms_per_row": round(per_row_ms, 3),
            "accuracy_eval_pool": round(acc, 4),
            "mean_max_prob": round(float(proba.max(axis=1).mean()), 4),
            **info,
        }
        print(json.dumps(rec, indent=2))
        summary.append(rec)

    (out_dir / "providers_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {out_dir}/providers_summary.json")

    if args.run_compare:
        specs = []
        for rec in summary:
            specs += ["--predictions", f"{rec['name']}={rec['path']}"]
        cmd = [
            sys.executable,
            "-m",
            "decisionops.cli.main",
            "compare",
            str(pack_dir / "dataset.jsonl"),
            *specs,
            "--cal-ratio",
            str(args.cal_ratio),
            "--seed",
            str(args.seed),
            "--output",
            str(pack_dir / "compare"),
        ]
        print("\n=== running compare ===")
        subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
