#!/usr/bin/env python3
"""
Run the local Laya decision model over an existing DecisionOps pack and write
predictions.jsonl, so it can be gated and compared against other providers.

Laya needs no training here: choices are supplied per request, so this is a
zero-shot run of the published checkpoint.

Usage:
  uv sync --extra laya
  uv run python scripts/export_laya_predictions.py --pack banking77 --limit 800
  uv run python scripts/export_laya_predictions.py --pack banking77 --limit 800 --no-builtin-temperature
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from decisionops.adapters.laya import LayaProvider
from decisionops.io import load_dataset
from decisionops.schemas import DecisionCase

ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    """Repo-relative path for committed stats, so they carry no local paths."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
PACK_DIRS = {"banking77": "banking77", "clinc": "clinc_oos", "toy": "toy-routing"}


def _subsample(cases: list[DecisionCase], limit: int | None, seed: int) -> list[DecisionCase]:
    if limit is None or limit >= len(cases):
        return cases
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = sorted(int(i) for i in rng.choice(len(cases), size=limit, replace=False))
    return [cases[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=sorted(PACK_DIRS), default="banking77")
    parser.add_argument(
        "--limit",
        type=int,
        default=800,
        help="Evaluate a random subset (CPU/MPS inference is slow). 0 = all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda")
    parser.add_argument(
        "--subdir",
        default=None,
        help="Variant inside the Laya repo, e.g. typed-decisions",
    )
    parser.add_argument(
        "--no-builtin-temperature",
        action="store_true",
        help="Skip Laya's shipped post-hoc temperature (report unscaled softmax).",
    )
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--examples-root", type=Path, default=ROOT / "examples")
    args = parser.parse_args()

    pack_dir = args.examples_root / PACK_DIRS[args.pack]
    dataset_path = pack_dir / "dataset.jsonl"
    if not dataset_path.exists():
        raise SystemExit(f"{dataset_path} not found — build the pack first.")

    cases = load_dataset(dataset_path)
    limit = None if args.limit in (0, None) else args.limit
    subset = _subsample(cases, limit, args.seed)
    n_choices = len(subset[0].choices)
    print(
        f"pack={args.pack}  cases={len(subset)}/{len(cases)}  choices={n_choices}"
    )

    provider = LayaProvider(
        device=args.device,
        subdir=args.subdir,
        apply_builtin_temperature=not args.no_builtin_temperature,
    )
    print(f"device={provider.device}  model={provider.model_name}")

    t0 = time.perf_counter()
    preds = provider.predict_many(
        subset, batch_size=args.batch_size, progress_every=5
    )
    elapsed = time.perf_counter() - t0

    suffix = "" if not args.no_builtin_temperature else "_rawT"
    name = args.out_name or f"laya{suffix}"
    out_dir = pack_dir / "providers"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{name}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.model_dump(mode="json"), ensure_ascii=False) + "\n")

    # Matching dataset subset so `gate`/`compare` can join on id without gaps.
    subset_path = out_dir / f"{name}_dataset.jsonl"
    with subset_path.open("w", encoding="utf-8") as f:
        for c in subset:
            f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")

    label_by_id = {c.id: c.label for c in subset}
    correct = sum(1 for p in preds if p.prediction == label_by_id[p.id])
    mean_conf = sum(p.confidence for p in preds) / len(preds)
    temps = {p.metadata.get("temperature_applied") for p in preds}
    stats = {
        "pack": args.pack,
        "n": len(preds),
        "n_choices": n_choices,
        "accuracy": round(correct / len(preds), 4),
        "mean_max_prob": round(mean_conf, 4),
        "temperature_applied": sorted(t for t in temps if t is not None),
        "builtin_temperature": not args.no_builtin_temperature,
        "seconds_total": round(elapsed, 1),
        "ms_per_case": round(elapsed / len(preds) * 1000, 1),
        "predictions": _rel(pred_path),
        "dataset_subset": _rel(subset_path),
    }
    (out_dir / f"{name}_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(
        f"\nNext:\n"
        f"  uv run decisionops gate {subset_path} \\\n"
        f"    --predictions {pred_path} --output {pack_dir}/gate-{name}"
    )


if __name__ == "__main__":
    main()
