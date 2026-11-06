#!/usr/bin/env python3
"""
Binary outcome labels for python/cpython PRs: did this PR cause trouble after
merge? No hand-written rule decides the label; only later PRs do.

  incident   a later PR (merged within --window-days) reverts it, or says it
             fixes a regression from / after it
  ok         neither happened within the window

How a later PR points at the culprit:

- revert: PR references in the title / body ("#123", "(GH-123)", /pull/123),
  plus "This reverts commit <sha>" resolved to its PR (resolve_cpython_reverts.py)
- regression: a PR whose title says "regression" (fix, not "regression tests")
  pointing at a PR, or at an issue (gh-NNN) that is itself the issue an earlier
  PR in the window was filed under. Issue matches are weaker: the issue may
  carry several PRs, so they are recorded as `regression_issue` and kept apart.

Dropped from the case pool: backports ("[3.x] ..."), PRs that are themselves
reverts, and PRs whose window has not closed by the last evidence date.

  uv run python scripts/label_cpython_incidents.py --prs data/cpython/full/prs.jsonl \
    --reverts data/cpython/full/reverts.json --out data/cpython/full/incidents.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PR_REF = re.compile(r"(?:#|\((?:GH|gh)-|/pull/)(\d{5,6})\b")
ISSUE_PREFIX = re.compile(r"^\s*(?:gh|GH|bpo)-(\d+)\b")
ISSUE_ANY = re.compile(r"\b(?:gh|GH)-(\d{5,6})\b")
REVERT_TITLE = re.compile(r"^\s*(?:(?:gh|GH)-\d+:\s*)?revert\b", re.I)
REGRESSION_FIX = re.compile(r"\bregression\b(?!\s+tests?\b)", re.I)
BACKPORT = re.compile(r"^\s*\[\d\.\d+\]")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _pr_refs(pr: dict) -> set[int]:
    return {int(m) for m in PR_REF.findall(f"{pr['title']}\n{pr['body']}")} - {pr["number"]}


def build(prs: list[dict], reverts: dict[int, list[int]], window_days: int):
    by_num = {p["number"]: p for p in prs}
    by_issue: dict[int, list[int]] = defaultdict(list)
    for p in prs:
        m = ISSUE_PREFIX.match(p["title"])
        if m:
            by_issue[int(m.group(1))].append(p["number"])

    hits: dict[int, list[tuple[str, int]]] = defaultdict(list)
    window = timedelta(days=window_days)

    def _add(culprit: int, kind: str, by: dict) -> None:
        c = by_num.get(culprit)
        if c is None or c["number"] == by["number"]:
            return
        lag = _ts(by["mergedAt"]) - _ts(c["mergedAt"])
        if timedelta(0) <= lag <= window:
            hits[culprit].append((kind, by["number"]))

    for p in prs:
        if REVERT_TITLE.search(p["title"]):
            targets = set(reverts.get(p["number"], [])) | _pr_refs(p)
            for n in targets:
                _add(n, "reverted", p)
        elif REGRESSION_FIX.search(p["title"]):
            refs = _pr_refs(p)
            for n in refs:
                _add(n, "regression_pr", p)
            if not refs:
                own = ISSUE_PREFIX.match(p["title"])
                for issue in {int(i) for i in ISSUE_ANY.findall(f"{p['title']} {p['body']}")}:
                    if own and issue == int(own.group(1)):
                        continue  # the fix's own issue, not the culprit's
                    for n in by_issue.get(issue, []):
                        _add(n, "regression_issue", p)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", type=Path, required=True)
    parser.add_argument("--reverts", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prs = [json.loads(line) for line in args.prs.open()]
    reverts = {int(k): v for k, v in json.loads(args.reverts.read_text()).items()}
    hits = build(prs, reverts, args.window_days)
    last = max(_ts(p["mergedAt"]) for p in prs)

    dropped: Counter = Counter()
    n_out = 0
    with args.out.open("w", encoding="utf-8") as f:
        for p in prs:
            if BACKPORT.match(p["title"]):
                dropped["backport"] += 1
                continue
            if REVERT_TITLE.search(p["title"]):
                dropped["is_revert"] += 1
                continue
            if _ts(p["mergedAt"]) + timedelta(days=args.window_days) > last:
                dropped["window_open"] += 1
                continue
            h = hits.get(p["number"], [])
            kinds = sorted({k for k, _ in h})
            f.write(
                json.dumps(
                    {
                        "number": p["number"],
                        "mergedAt": p["mergedAt"],
                        "incident": bool(h),
                        "kinds": kinds,
                        "by": sorted({b for _, b in h}),
                    }
                )
                + "\n"
            )
            n_out += 1
    print(f"wrote {n_out} labelled PRs; dropped {dict(dropped)}")


if __name__ == "__main__":
    main()
