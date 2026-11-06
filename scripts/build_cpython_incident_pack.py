#!/usr/bin/env python3
"""
Build the cpython incident pack: every labelled incident PR (minus planned,
temporary reverts) plus a time-random sample of ok PRs, in the same `state`
layout as examples/code-review.

The sample is enriched: incidents are kept in full, ok PRs are subsampled, so
the pack's incident rate is far above the population rate. `metadata.weight`
is the inverse sampling rate of each row's class; anything that reports a rate
for the population must use it.

  uv run python scripts/build_cpython_incident_pack.py \
    --prs data/cpython/full/prs.jsonl --incidents data/cpython/full/incidents.jsonl \
    --n-ok 3000 --out data/cpython/full/pack.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

CHOICES = ["ok", "incident"]
QUESTION = (
    "Will this pull request to CPython have to be reverted, or cause a regression "
    "that needs a follow-up fix, within 90 days of merging?"
)
# Reverts planned at merge time ("temporarily disable X") are not incidents.
PLANNED = re.compile(r"\btemporar|\bdisable\b|\bskip\b", re.I)


def _state_of():
    path = Path(__file__).with_name("label_cpython_prs.py")
    spec = importlib.util.spec_from_file_location("label_cpython_prs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.state_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--n-ok", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    state_of = _state_of()
    prs = {p["number"]: p for p in map(json.loads, args.prs.open())}
    labelled = [json.loads(line) for line in args.incidents.open()]
    inc = [r for r in labelled if r["incident"] and not PLANNED.search(prs[r["number"]]["title"])]
    planned = sum(r["incident"] and PLANNED.search(prs[r["number"]]["title"]) is not None
                  for r in labelled)
    ok = [r for r in labelled if not r["incident"]]
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(ok), size=min(args.n_ok, len(ok)), replace=False))
    ok_sample = [ok[i] for i in pick]
    w_inc, w_ok = 1.0, len(ok) / len(ok_sample)

    rows = []
    for r, label, w in [(r, "incident", w_inc) for r in inc] + [
        (r, "ok", w_ok) for r in ok_sample
    ]:
        p = prs[r["number"]]
        rows.append(
            {
                "id": f"cpython-{p['number']}",
                "state": state_of(p),
                "question": QUESTION,
                "choices": list(CHOICES),
                "label": label,
                "metadata": {
                    "repo": "python/cpython",
                    "number": p["number"],
                    "merged_at": p["mergedAt"],
                    "incident_kinds": r["kinds"],
                    "incident_by": r["by"],
                    "weight": w,
                },
            }
        )
    rows.sort(key=lambda x: x["metadata"]["merged_at"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pop = len(inc) + len(ok)
    print(
        f"wrote {len(rows)} cases: {len(inc)} incident (dropped {planned} planned), "
        f"{len(ok_sample)} ok of {len(ok)}; pack rate {len(inc) / len(rows):.2%}, "
        f"population rate {len(inc) / pop:.2%}, ok weight {w_ok:.3f}"
    )


if __name__ == "__main__":
    main()
