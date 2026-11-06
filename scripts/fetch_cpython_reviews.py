#!/usr/bin/env python3
"""
Fetch the review each python/cpython PR actually received: review states and
authors, inline review comments, conversation comments, commits. Read-only,
public data, `gh api graphql` (needs `gh auth login`).

  uv run python scripts/fetch_cpython_reviews.py --prs data/cpython/prs.jsonl \
    --out data/cpython/reviews.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BATCH = 25
FIELDS = """
  number createdAt mergedAt authorAssociation
  author { login } mergedBy { login }
  commits { totalCount }
  comments { totalCount }
  reviewThreads { totalCount }
  reviews(first: 100) {
    totalCount
    nodes { state authorAssociation author { login } comments { totalCount } }
  }
"""


def _query(numbers: list[int]) -> str:
    parts = [f"p{n}: pullRequest(number: {n}) {{ {FIELDS} }}" for n in numbers]
    return '{ repository(owner: "python", name: "cpython") { ' + " ".join(parts) + " } }"


def _run(numbers: list[int]) -> dict:
    for attempt in range(5):
        proc = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={_query(numbers)}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)["data"]["repository"]
        time.sleep(3 * (attempt + 1))
    raise SystemExit(f"gh api graphql failed: {proc.stderr[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    numbers = [json.loads(line)["number"] for line in args.prs.open()]
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(0, len(numbers), BATCH):
            chunk = numbers[i : i + BATCH]
            data = _run(chunk)
            for n in chunk:
                node = data.get(f"p{n}")
                if node is not None:
                    f.write(json.dumps(node, ensure_ascii=False) + "\n")
            print(f"  {min(i + BATCH, len(numbers))}/{len(numbers)}", file=sys.stderr, flush=True)
    print(f"wrote reviews for {len(numbers)} PRs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
