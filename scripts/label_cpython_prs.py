#!/usr/bin/env python3
"""
Label fetched python/cpython PRs with the review depth they turned out to need,
from after-the-fact evidence, and write a DecisionOps dataset.

Evidence, strongest first (each label records which rule produced it):

  human  security       `type-security` label, or a Misc/NEWS.d/**/Security news entry
  deep   reverted       a later PR reverts it (title "Revert ..." naming this PR,
                        or body "This reverts commit <merge sha>")
  deep   regression     a later PR says it fixes a regression from / after this PR
  ---- no adverse outcome within the maturity window; level from what it touches:
  skip   docs_only      only Doc/, *.md, *.rst outside Misc/NEWS.d, .github/ docs
  shallow tests_only    only Lib/test/, Tools/, CI config, Misc/ (incl. news)
  shallow small_code    touches shipped code, <= 20 changed lines, no adverse outcome
  deep   code           touches shipped code, > 20 changed lines, no adverse outcome

The rules below the line are a judgement, not an outcome: they say "nothing went
wrong, so the review it got was enough". Reports must keep them separate.

Backports ("[3.x] ...") and PRs that are themselves reverts are dropped: a
backport repeats a reviewed change, and a revert is the evidence, not a case.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

LEVELS = ["skip", "shallow", "deep", "human"]
QUESTION = "How much review does this pull request need before merging?"
SMALL_CODE_LINES = 20

# PR references only: "#123456", "(GH-123456)" / "(gh-123456)" as cherry-pick
# and merge titles write them, and pull URLs. A bare "gh-123456:" prefix is an
# issue number, not a PR, and must not count.
PR_REF = re.compile(r"(?:#|\((?:GH|gh)-|/pull/)(\d{5,6})\b")
REVERT_TITLE = re.compile(r"^\s*(?:gh-\d+:\s*)?revert\b", re.I)
REGRESSION = re.compile(r"\bregression\b", re.I)


CODE_DIRS = ("Lib/", "Modules/", "Objects/", "Python/", "Include/", "Parser/", "PC/", "PCbuild/")


def _is_doc(path: str) -> bool:
    """Documentation only: cannot change shipped behaviour."""
    if path.startswith("Misc/NEWS.d/"):
        return False
    if path.startswith("Doc/"):
        return True
    return path.endswith((".md", ".rst", ".txt")) and not path.startswith(CODE_DIRS)


def _is_support(path: str) -> bool:
    return path.startswith(
        ("Lib/test/", "Tools/", ".github/", "Misc/", ".azure-pipelines/",
         "Lib/idlelib/idle_test/")
    ) or path in {".pre-commit-config.yaml", ".gitignore", ".gitattributes"}


def _referenced_prs(pr: dict) -> set[int]:
    text = f"{pr['title']}\n{pr['body']}"
    return {int(m) for m in PR_REF.findall(text) if int(m) != pr["number"]}


def outcome_evidence(evidence: list[dict]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Map PR number -> later PRs that revert it / fix a regression it caused."""
    reverted: dict[int, list[int]] = {}
    regressed: dict[int, list[int]] = {}
    for pr in evidence:
        refs = _referenced_prs(pr)
        if REVERT_TITLE.search(pr["title"]):
            for n in refs:
                reverted.setdefault(n, []).append(pr["number"])
        elif REGRESSION.search(pr["title"]) or re.search(
            r"regression (?:from|in|after|introduced)", pr["body"], re.I
        ):
            for n in refs:
                regressed.setdefault(n, []).append(pr["number"])
    return reverted, regressed


def label(pr: dict, reverted: dict, regressed: dict) -> tuple[str, str]:
    files = pr["files"]
    if "type-security" in pr["labels"] or any(
        f.startswith("Misc/NEWS.d/") and "/Security/" in f for f in files
    ):
        return "human", "security"
    if pr["number"] in reverted:
        return "deep", "reverted"
    if pr["number"] in regressed:
        return "deep", "regression"
    if files and all(_is_doc(f) for f in files):
        return "skip", "docs_only"
    if files and all(_is_doc(f) or _is_support(f) for f in files):
        return "shallow", "tests_only"
    if pr["additions"] + pr["deletions"] <= SMALL_CODE_LINES:
        return "shallow", "small_code"
    return "deep", "code"


def state_of(pr: dict, max_files: int = 40, max_desc: int = 600) -> str:
    """Same layout as examples/code-review and examples/github-action/pr_route.py."""
    files = pr["files"]
    body = " ".join(re.sub(r"<!--.*?-->", " ", pr["body"], flags=re.S).split())
    body = body if len(body) <= max_desc else body[: max_desc - 1] + "…"
    shown = ", ".join(files[:max_files])
    more = f" (+{len(files) - max_files} more)" if len(files) > max_files else ""
    return (
        f"PR title: {' '.join(pr['title'].split())}\n"
        f"Description: {body or '(none)'}\n"
        f"Files changed ({pr['changedFiles']}): {shown}{more}\n"
        f"Lines: +{pr['additions']} -{pr['deletions']}\n"
        f"Author: {pr['author'] or 'unknown'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--reverts",
        type=Path,
        default=None,
        help="revert PR -> reverted PRs, from resolve_cpython_reverts.py (resolves SHAs)",
    )
    parser.add_argument("--maturity-days", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prs = [json.loads(line) for line in args.prs.open()]
    evidence = [json.loads(line) for line in args.evidence.open()]
    reverted, regressed = outcome_evidence(evidence)
    if args.reverts is not None:
        for rev, targets in json.loads(args.reverts.read_text()).items():
            for n in targets:
                if int(rev) not in reverted.setdefault(n, []):
                    reverted[n].append(int(rev))
    last_evidence = max(
        datetime.fromisoformat(p["mergedAt"].replace("Z", "+00:00")) for p in evidence
    )

    rows, dropped = [], Counter()
    for pr in prs:
        merged = datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00"))
        if re.match(r"^\s*\[\d\.\d+\]", pr["title"]):
            dropped["backport"] += 1
            continue
        if REVERT_TITLE.search(pr["title"]):
            dropped["is_revert"] += 1
            continue
        if merged + timedelta(days=args.maturity_days) > last_evidence:
            dropped["immature"] += 1
            continue
        level, source = label(pr, reverted, regressed)
        rows.append(
            {
                "id": f"cpython-{pr['number']}",
                "state": state_of(pr),
                "question": QUESTION,
                "choices": list(LEVELS),
                "label": level,
                "metadata": {
                    "domain": "code-review",
                    "repo": "python/cpython",
                    "number": pr["number"],
                    "merged_at": pr["mergedAt"],
                    "label_source": source,
                    "evidence_prs": reverted.get(pr["number"], [])
                    + regressed.get(pr["number"], []),
                    "gh_labels": pr["labels"],
                },
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases to {args.out}; dropped {dict(dropped)}")
    print("labels", dict(Counter(r["label"] for r in rows)))
    print("sources", dict(Counter(r["metadata"]["label_source"] for r in rows)))


if __name__ == "__main__":
    main()
