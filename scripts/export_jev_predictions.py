#!/usr/bin/env python3
"""
Run hosted Jev over an existing DecisionOps pack and write predictions.jsonl,
so it can be gated and compared against other providers.

`compare` is offline-only, so live providers are exported first. Each case is
one API call: start with a small --limit to check cost and latency.

Usage:
  uv sync --extra jev
  export TYPESAFE_API_KEY=...
  uv run python scripts/export_jev_predictions.py --pack toy --limit 20
  uv run python scripts/export_jev_predictions.py --pack banking77 --limit 200 \
    --contract examples/banking77/contract.json --model my-finetuned-jev
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from decisionops.adapters.jev import DEFAULT_MODEL, JevProvider
from decisionops.contract import load_contract
from decisionops.io import load_dataset
from decisionops.schemas import DecisionCase

ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--pack", choices=sorted(PACK_DIRS), default="toy")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Evaluate a random subset (each case is one paid API call). 0 = all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Jev model id, or a fine-tuned id")
    parser.add_argument("--base-url", default=None, help="Private / custom Jev endpoint")
    parser.add_argument(
        "--question",
        choices=["choice", "noul"],
        default="choice",
        help="noul = calibrated yes/no; needs a contract with yes_choice",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Decision contract whose descriptions / instructions become the model input",
    )
    parser.add_argument("--out-name", default="jev")
    parser.add_argument("--examples-root", type=Path, default=ROOT / "examples")
    args = parser.parse_args()

    pack_dir = args.examples_root / PACK_DIRS[args.pack]
    dataset_path = pack_dir / "dataset.jsonl"
    if not dataset_path.exists():
        raise SystemExit(f"{dataset_path} not found — build the pack first.")

    cases = load_dataset(dataset_path)
    limit = None if args.limit in (0, None) else args.limit
    subset = _subsample(cases, limit, args.seed)
    print(f"pack={args.pack}  cases={len(subset)}/{len(cases)}  choices={len(subset[0].choices)}")

    contract = load_contract(args.contract) if args.contract else None
    yes_choice = contract.yes_choice if contract else None
    if args.question == "noul" and yes_choice is None:
        raise SystemExit("--question noul needs --contract with yes_choice")
    provider = JevProvider(
        model=args.model,
        base_url=args.base_url,
        descriptions=contract.descriptions if contract else None,
        contract_instructions=contract.instructions if contract else None,
        question_type=args.question,
        yes_choice=yes_choice,
    )

    t0 = time.perf_counter()
    preds = provider.predict_many(subset, progress_every=10)
    elapsed = time.perf_counter() - t0

    out_dir = pack_dir / "providers"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{args.out_name}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.model_dump(mode="json"), ensure_ascii=False) + "\n")

    # Matching dataset subset so `gate`/`compare` can join on id without gaps.
    subset_path = out_dir / f"{args.out_name}_dataset.jsonl"
    with subset_path.open("w", encoding="utf-8") as f:
        for c in subset:
            f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")

    label_by_id = {c.id: c.label for c in subset}
    correct = sum(1 for p in preds if p.prediction == label_by_id[p.id])
    stats = {
        "pack": args.pack,
        "n": len(preds),
        "model_requested": args.model,
        "models_served": sorted({p.model for p in preds if p.model}),
        "accuracy": round(correct / len(preds), 4),
        "mean_max_prob": round(sum(p.confidence for p in preds) / len(preds), 4),
        "contract": contract.summary() if contract else None,
        "seconds_total": round(elapsed, 1),
        "ms_per_case": round(elapsed / len(preds) * 1000, 1),
        "predictions": str(pred_path),
        "dataset_subset": str(subset_path),
    }
    (out_dir / f"{args.out_name}_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(
        f"\nNext:\n"
        f"  uv run decisionops gate {subset_path} \\\n"
        f"    --predictions {pred_path} --output {pack_dir}/gate-{args.out_name}"
    )


if __name__ == "__main__":
    main()
