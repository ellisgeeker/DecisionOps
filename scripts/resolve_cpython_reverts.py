#!/usr/bin/env python3
"""
Resolve "This reverts commit <sha>" in cpython revert PRs to the PR that
introduced that commit, via the commit's associated pull requests. Read-only.

Writes {revert_pr: [reverted_pr, ...]} for every revert PR in the evidence file,
combining PR references in the title/body with resolved SHAs.

  uv run python scripts/resolve_cpython_reverts.py \
    --evidence data/cpython/evidence_prs.jsonl --out data/cpython/reverts.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

SHA = re.compile(r"\breverts? (?:commit )?([0-9a-f]{7,40})\b", re.I)


def _labeller():
    path = Path(__file__).with_name("label_cpython_prs.py")
    spec = importlib.util.spec_from_file_location("label_cpython_prs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prs_for_commit(sha: str) -> list[int]:
    proc = subprocess.run(
        ["gh", "api", f"repos/python/cpython/commits/{sha}/pulls", "--jq", ".[].number"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [int(x) for x in proc.stdout.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lab = _labeller()
    out: dict[str, list[int]] = {}
    for line in args.evidence.open():
        pr = json.loads(line)
        if not lab.REVERT_TITLE.search(pr["title"]):
            continue
        targets = set(lab._referenced_prs(pr))
        for sha in SHA.findall(pr["body"]):
            targets |= set(prs_for_commit(sha))
        targets.discard(pr["number"])
        out[str(pr["number"])] = sorted(targets)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    unresolved = [k for k, v in out.items() if not v]
    print(f"{len(out)} revert PRs; {len(unresolved)} unresolved: {unresolved}")


if __name__ == "__main__":
    main()
