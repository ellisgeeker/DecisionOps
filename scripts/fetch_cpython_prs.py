#!/usr/bin/env python3
"""
Fetch merged python/cpython PRs into main for a date window, plus the evidence
used to label them after the fact. Read-only, public data, GitHub GraphQL via
`gh api graphql` (needs `gh auth login`; no write scopes used).

Output (JSONL, one PR per line): number, title, body, mergedAt, additions,
deletions, files, labels, author, and the PR numbers that later reference it
as reverted or as the cause of a regression (filled by label_cpython_prs.py).

  uv run python scripts/fetch_cpython_prs.py --since 2025-01-01 --until 2025-03-31 \
    --out data/cpython/prs.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

QUERY = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 50, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number title body mergedAt additions deletions changedFiles
        baseRefName isCrossRepository
        author { login }
        labels(first: 20) { nodes { name } }
        files(first: 100) { nodes { path additions deletions } }
      }
    }
  }
}
"""


def gh_graphql(q: str, cursor: str | None) -> dict:
    args = ["gh", "api", "graphql", "-f", f"query={QUERY}", "-F", f"q={q}"]
    if cursor:
        args += ["-F", f"cursor={cursor}"]
    for attempt in range(5):
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)["data"]["search"]
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"gh api graphql failed: {proc.stderr[:500]}")


def fetch_window(since: str, until: str, step_days: int = 1) -> list[dict]:
    # GitHub search returns at most 1,000 results per query, so walk in slices
    # small enough to stay under it (cpython merges ~100 PRs a week into main).
    from datetime import date, timedelta

    out: list[dict] = []
    day = date.fromisoformat(since)
    end = date.fromisoformat(until)
    while day <= end:
        last = min(day + timedelta(days=step_days - 1), end)
        q = (
            "repo:python/cpython is:pr is:merged base:main "
            f"merged:{day.isoformat()}..{last.isoformat()}"
        )
        cursor = None
        while True:
            page = gh_graphql(q, cursor)
            if cursor is None and page["issueCount"] > 1000:
                raise SystemExit(f"{q}: {page['issueCount']} results > 1000; lower --step-days")
            for n in page["nodes"]:
                if not n:
                    continue
                out.append(
                    {
                        "number": n["number"],
                        "title": n["title"],
                        "body": n["body"] or "",
                        "mergedAt": n["mergedAt"],
                        "additions": n["additions"],
                        "deletions": n["deletions"],
                        "changedFiles": n["changedFiles"],
                        "author": (n.get("author") or {}).get("login"),
                        "labels": [lab["name"] for lab in n["labels"]["nodes"]],
                        "files": [f["path"] for f in n["files"]["nodes"]],
                    }
                )
            if not page["pageInfo"]["hasNextPage"]:
                break
            cursor = page["pageInfo"]["endCursor"]
        print(f"  {day}..{last}: total {len(out)}", file=sys.stderr, flush=True)
        day = last + timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = fetch_window(args.since, args.until, args.step_days)
    rows.sort(key=lambda r: r["mergedAt"])
    seen: set[int] = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["number"] in seen:
                continue
            seen.add(r["number"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(seen)} PRs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
