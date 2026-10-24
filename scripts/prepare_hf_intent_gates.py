#!/usr/bin/env python3
"""
Download public intent datasets from Hugging Face, train a sklearn
TF-IDF + LogisticRegression head, and export DecisionOps JSONL pairs.

Supported packs:
  - banking77  → mteb/banking77 (parquet mirror of BANKING77)
  - clinc      → clinc/clinc_oos (config: plus|small|imbalanced)

Examples:
  uv sync --extra bench
  uv run python scripts/prepare_hf_intent_gates.py --pack banking77
  uv run python scripts/prepare_hf_intent_gates.py --pack clinc
  uv run python scripts/prepare_hf_intent_gates.py --pack all
  uv run python scripts/prepare_hf_intent_gates.py --pack all --run-gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    """Repo-relative path for committed stats, so they carry no local paths."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `datasets`.\n"
            "Install with:  uv sync --extra bench"
        ) from exc
    from datasets import load_dataset

    return load_dataset


def _label_names_from_feature(feature) -> list[str] | None:
    """Best-effort extract ClassLabel names from a datasets Feature."""
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    # datasets 2.x ClassLabel sometimes nested
    inner = getattr(feature, "feature", None)
    names = getattr(inner, "names", None)
    if names:
        return list(names)
    return None


def _as_str_label(raw, names: list[str] | None) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (int, np.integer)):
        if names is not None and 0 <= int(raw) < len(names):
            return str(names[int(raw)])
        return str(int(raw))
    return str(raw)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _train_and_export(
    *,
    pack_name: str,
    texts_train: list[str],
    labels_train: list[str],
    texts_eval: list[str],
    labels_eval: list[str],
    ids_eval: list[str],
    splits_eval: list[str],
    question: str,
    domain: str,
    out_dir: Path,
    seed: int,
    C: float,
) -> dict:
    """
    Fit on train only; export DecisionOps JSONL for the eval pool only
    (HF test/validation) so gate metrics are not inflated by train leakage.
    """
    choices = sorted(set(labels_train) | set(labels_eval))
    if len(choices) < 2:
        raise RuntimeError(f"{pack_name}: need >= 2 classes, got {choices!r}")

    clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    C=C,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(texts_train, labels_train)
    classes = list(clf.named_steps["lr"].classes_)
    proba = clf.predict_proba(texts_eval)

    cases: list[dict] = []
    preds: list[dict] = []
    for i, text in enumerate(texts_eval):
        probs = {str(c): float(proba[i, j]) for j, c in enumerate(classes)}
        # Classes missing from the linear head (never seen in train) get 0 mass.
        for ch in choices:
            probs.setdefault(ch, 0.0)
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        pred = max(probs, key=probs.get)
        cid = ids_eval[i]
        cases.append(
            {
                "id": cid,
                "state": text,
                "question": question,
                "choices": choices,
                "label": labels_eval[i],
                "metadata": {
                    "domain": domain,
                    "source_split": splits_eval[i],
                    "pack": pack_name,
                },
            }
        )
        preds.append(
            {
                "id": cid,
                "prediction": pred,
                "probabilities": probs,
                "latency_ms": None,
                "provider": "sklearn",
                "model": f"tfidf-lr-C{C}",
                "metadata": {"pack": pack_name},
            }
        )

    _write_jsonl(out_dir / "dataset.jsonl", cases)
    _write_jsonl(out_dir / "predictions.jsonl", preds)

    acc = float(np.mean([p["prediction"] == c["label"] for p, c in zip(preds, cases)]))
    mean_conf = float(np.mean([max(p["probabilities"].values()) for p in preds]))
    stats = {
        "pack": pack_name,
        "n_eval": len(cases),
        "n_classes": len(choices),
        "n_train_fit": len(texts_train),
        "accuracy_eval": acc,
        "mean_max_prob": mean_conf,
        "out_dir": _rel(out_dir),
        "protocol": "fit_train_export_heldout",
    }
    (out_dir / "build_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def prepare_banking77(out_dir: Path, seed: int, C: float) -> dict:
    """
    BANKING77 via mteb/banking77 (parquet).

    PolyAI/banking77 still ships a legacy dataset script that recent
    `datasets` versions refuse to execute.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("mteb/banking77")
    label_feat = ds["train"].features.get("label")
    names = _label_names_from_feature(label_feat)

    texts_train, labels_train = [], []
    texts_eval, labels_eval, ids_eval, splits_eval = [], [], [], []

    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        for i, row in enumerate(ds[split_name]):
            text = str(row["text"]).strip()
            if row.get("label_text"):
                lab = str(row["label_text"])
            else:
                lab = _as_str_label(row["label"], names)
            if split_name == "train":
                texts_train.append(text)
                labels_train.append(lab)
            else:
                cid = f"banking77-{split_name}-{i:05d}"
                texts_eval.append(text)
                labels_eval.append(lab)
                ids_eval.append(cid)
                splits_eval.append(split_name)

    return _train_and_export(
        pack_name="banking77",
        texts_train=texts_train,
        labels_train=labels_train,
        texts_eval=texts_eval,
        labels_eval=labels_eval,
        ids_eval=ids_eval,
        splits_eval=splits_eval,
        question="Which banking support intent should we route this to?",
        domain="banking-intent",
        out_dir=out_dir,
        seed=seed,
        C=C,
    )


def prepare_clinc(out_dir: Path, seed: int, C: float, config: str = "plus") -> dict:
    load_dataset = _require_datasets()
    ds = load_dataset("clinc/clinc_oos", config)

    sample = ds["train"][0]
    intent_key = "intent" if "intent" in sample else None
    text_key = "text" if "text" in sample else ("query" if "query" in sample else None)
    if text_key is None:
        raise RuntimeError(f"clinc: cannot find text field in {list(sample)}")

    label_feat = ds["train"].features.get("intent") or ds["train"].features.get("label")
    names = _label_names_from_feature(label_feat)

    texts_train, labels_train = [], []
    texts_eval, labels_eval, ids_eval, splits_eval = [], [], [], []

    fit_split = "train" if "train" in ds else next(iter(ds.keys()))
    # Held-out pool for DecisionOps export (never used to fit the head).
    eval_splits = [
        s
        for s in ("validation", "val", "test", "oos_test", "oos_validation", "oos_val")
        if s in ds
    ]
    if not eval_splits:
        raise RuntimeError("clinc: no held-out splits found")

    def _label_of(row) -> str:
        if intent_key and isinstance(row.get(intent_key), str):
            lab = str(row[intent_key])
        else:
            raw = row.get("intent", row.get("label"))
            lab = _as_str_label(raw, names)
        if lab.lower() in {"oos", "out_of_scope", "out-of-scope"}:
            return "oos"
        return lab

    for i, row in enumerate(ds[fit_split]):
        texts_train.append(str(row[text_key]).strip())
        labels_train.append(_label_of(row))

    for split_name in eval_splits:
        for i, row in enumerate(ds[split_name]):
            text = str(row[text_key]).strip()
            lab = _label_of(row)
            cid = f"clinc-{config}-{split_name}-{i:05d}"
            texts_eval.append(text)
            labels_eval.append(lab)
            ids_eval.append(cid)
            splits_eval.append(split_name)

    return _train_and_export(
        pack_name=f"clinc_oos_{config}",
        texts_train=texts_train,
        labels_train=labels_train,
        texts_eval=texts_eval,
        labels_eval=labels_eval,
        ids_eval=ids_eval,
        splits_eval=splits_eval,
        question="Which assistant intent should we route this to? (oos = out of scope)",
        domain="assistant-intent",
        out_dir=out_dir,
        seed=seed,
        C=C,
    )


def run_gate_on(out_dir: Path, seed: int, cal_ratio: float) -> None:
    # Import from installed package
    sys.path.insert(0, str(ROOT / "src"))
    from decisionops.eval import run_gate
    from decisionops.io import load_dataset, load_predictions
    from decisionops.report import print_gate_report, write_json_report

    cases = load_dataset(out_dir / "dataset.jsonl")
    preds = load_predictions(out_dir / "predictions.jsonl")
    report = run_gate(
        cases,
        preds,
        cal_ratio=cal_ratio,
        seed=seed,
        calibrate=True,
        provider="sklearn",
    )
    print_gate_report(report)
    report_dir = out_dir / "gate"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = write_json_report(report, report_dir / "gate_report.json")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        choices=["banking77", "clinc", "all"],
        default="all",
    )
    parser.add_argument("--clinc-config", default="plus", help="clinc/clinc_oos config name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization")
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=ROOT / "examples",
    )
    parser.add_argument("--run-gate", action="store_true")
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    args = parser.parse_args()

    packs: list[str] = (
        ["banking77", "clinc"] if args.pack == "all" else [args.pack]
    )
    all_stats = []

    for pack in packs:
        if pack == "banking77":
            out = args.examples_root / "banking77"
            print(f"\n=== Preparing BANKING77 → {out} ===")
            stats = prepare_banking77(out, seed=args.seed, C=args.C)
        else:
            out = args.examples_root / "clinc_oos"
            print(f"\n=== Preparing CLINC OOS ({args.clinc_config}) → {out} ===")
            stats = prepare_clinc(
                out, seed=args.seed, C=args.C, config=args.clinc_config
            )
        print(json.dumps(stats, indent=2))
        all_stats.append(stats)
        if args.run_gate:
            print(f"\n=== Gate on {out.name} ===")
            run_gate_on(out, seed=args.seed, cal_ratio=args.cal_ratio)

    summary_path = args.examples_root / "hf_intent_build_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
