#!/usr/bin/env python3
"""
Score a DecisionOps dataset with live Jev, appending one prediction per line and
skipping ids already scored, so an interrupted run resumes where it stopped.
The question type follows the contract: noul (yes_choice), score (choice_order),
else choice.

  export TYPESAFE_API_KEY=...
  uv run python scripts/score_jev_resumable.py --dataset data/cpython/full/pack.jsonl \
    --contract examples/contracts/cpython-incident.json --out data/cpython/full/jev.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from decisionops.adapters.jev import DEFAULT_MODEL, JevProvider
from decisionops.contract import load_contract
from decisionops.io import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    if contract.yes_choice:
        question = "noul"
    elif contract.is_ordinal:
        question = "score"
    else:
        question = "choice"
    provider = JevProvider(
        model=args.model,
        question_type=question,
        yes_choice=contract.yes_choice,
        choice_order=contract.choice_order if contract.is_ordinal else None,
        descriptions=contract.descriptions,
        contract_instructions=contract.instructions,
    )
    cases = load_dataset(args.dataset)
    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open() if line.strip()}
    todo = [c for c in cases if c.id not in done]
    print(f"{question}: {len(done)} already scored, {len(todo)} to go", flush=True)

    t0 = time.perf_counter()
    failures = 0
    with args.out.open("a", encoding="utf-8") as f:
        for i, case in enumerate(todo, 1):
            try:
                pred = provider.predict(case)
            except (RuntimeError, ValueError) as exc:
                failures += 1
                print(f"  {case.id}: {exc}", flush=True)
                if failures > 20:
                    raise SystemExit("too many failures; stopping (rerun to resume)") from exc
                continue
            f.write(json.dumps(pred.model_dump(mode="json"), ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                rate = (time.perf_counter() - t0) / i
                print(f"  {i}/{len(todo)}  {rate * 1000:.0f} ms/case  "
                      f"eta {rate * (len(todo) - i) / 60:.1f} min", flush=True)
    print(f"done: {len(todo) - failures} scored, {failures} failed", flush=True)


if __name__ == "__main__":
    main()
