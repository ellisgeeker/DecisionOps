#!/usr/bin/env python3
"""
Build the synthetic code-review showcase pack: one PR per case, labelled with
the review depth it actually needed (skip < shallow < deep < human).

The label is not "what a reviewer guessed". It is derived from a simulated
outcome, the way the labelling guide says real labels are collected:

  no_issue            -> skip     (merged, nothing found, never reverted)
  style_comment       -> shallow  (quick review left only cosmetic comments)
  bug_in_deep_review  -> deep     (a deep review found a real defect)
  incident_or_revert  -> human    (shipped and caused an incident / revert, or
                                   policy requires a human sign-off)

About a fifth of cases are deliberately hard: the title or size points one way
and the substance the other (a "typo fix" that touches an auth check, a
40-file rename that is purely mechanical). Those are what make confidence
thresholds worth certifying.

Deterministic for a given --seed. No network, no model.

  uv run python scripts/build_code_review_pack.py --n 600 --out examples/code-review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = ["skip", "shallow", "deep", "human"]
OUTCOME = {
    "skip": "no_issue",
    "shallow": "style_comment",
    "deep": "bug_in_deep_review",
    "human": "incident_or_revert",
}
QUESTION = "How much review does this pull request need before merging?"

SERVICES = ["api", "web", "worker", "search", "notifications", "reports", "gateway"]
MODULES = ["orders", "users", "catalog", "inventory", "shipping", "analytics", "export"]
PEOPLE = ["dev-a", "dev-b", "dev-c", "dev-d", "dev-e", "dev-f"]

# (title template, body template, file templates, (min, max) lines changed)
# Placeholders: {svc} {mod} {n}
EASY: dict[str, list[tuple[str, str, list[str], tuple[int, int]]]] = {
    "skip": [
        ("Fix typo in {mod} README", "Corrects spelling in the setup section.",
         ["docs/{mod}/README.md"], (1, 4)),
        ("Update CONTRIBUTING link", "The wiki moved; points the link at the new page.",
         ["CONTRIBUTING.md"], (1, 2)),
        ("Reword comment in {mod} service", "Comment-only change, no code touched.",
         ["{svc}/{mod}/service.py"], (1, 6)),
        ("Run formatter on {mod} docs", "Whitespace and line wrapping only.",
         ["docs/{mod}/guide.md", "docs/{mod}/faq.md"], (10, 60)),
        ("Add {mod} entry to CHANGELOG", "Release notes for the last sprint.",
         ["CHANGELOG.md"], (3, 12)),
        ("Bump dev-only linter to next patch", "Pre-commit hook version; not shipped.",
         [".pre-commit-config.yaml"], (1, 2)),
    ],
    "shallow": [
        ("Rename helper in {mod}", "Renames `_fmt` to `_format_row`; callers updated.",
         ["{svc}/{mod}/utils.py", "{svc}/{mod}/views.py"], (6, 30)),
        ("Add unit tests for {mod} parser", "New tests only; no production change.",
         ["tests/{mod}/test_parser.py"], (40, 120)),
        ("Clarify log message in {svc}", "Adds the request id to an info-level log line.",
         ["{svc}/{mod}/handlers.py"], (1, 4)),
        ("Patch-bump {mod} client library", "Lockfile update, patch release, no API change.",
         ["poetry.lock", "pyproject.toml"], (4, 40)),
        ("Tweak staging config for {svc}", "Raises the staging log level; prod untouched.",
         ["deploy/staging/{svc}.yaml"], (1, 3)),
        ("Extract constant in {mod}", "Replaces a repeated literal with a named constant.",
         ["{svc}/{mod}/constants.py", "{svc}/{mod}/service.py"], (4, 16)),
    ],
    "deep": [
        ("Add pagination to {mod} list endpoint", "New cursor parameter and page size limit.",
         ["{svc}/{mod}/views.py", "{svc}/{mod}/queries.py", "tests/{mod}/test_views.py"],
         (60, 220)),
        ("Cache {mod} lookups in {svc}", "Adds an in-process TTL cache around the lookup.",
         ["{svc}/{mod}/cache.py", "{svc}/{mod}/service.py"], (40, 160)),
        ("Parallelise {mod} export job", "Splits the export into {n} worker threads.",
         ["worker/{mod}/export.py", "worker/{mod}/pool.py"], (80, 260)),
        ("Rewrite {mod} retry logic", "Exponential backoff with jitter; max {n} attempts.",
         ["{svc}/{mod}/retry.py", "tests/{mod}/test_retry.py"], (50, 180)),
        ("Change {mod} query to use join", "Replaces N+1 queries with a single join.",
         ["{svc}/{mod}/queries.py"], (20, 90)),
        ("Handle timeouts in {svc} client", "Catches timeouts and returns a partial result.",
         ["{svc}/{mod}/client.py", "{svc}/{mod}/errors.py"], (30, 120)),
    ],
    "human": [
        ("Change refund rounding in billing", "Rounds refunds to the nearest cent instead of down.",
         ["billing/refunds.py", "billing/money.py"], (8, 40)),
        ("Relax permission check on {mod} admin", "Lets support staff view {mod} records.",
         ["{svc}/auth/permissions.py", "{svc}/{mod}/admin.py"], (6, 30)),
        ("Drop legacy {mod} columns", "Migration removes {n} columns; irreversible.",
         ["migrations/0{n}_drop_{mod}_legacy.py"], (20, 80)),
        ("Rotate signing key handling", "Loads the JWT signing key from the new vault path.",
         ["{svc}/auth/keys.py", "deploy/prod/{svc}.yaml"], (10, 50)),
        ("Change prod autoscaling for {svc}", "Lowers min replicas from 6 to {n}.",
         ["deploy/prod/{svc}.yaml"], (2, 6)),
        ("Log full {mod} payload for debugging",
         "Writes request bodies, incl. email/phone, to logs.",
         ["{svc}/{mod}/handlers.py"], (2, 10)),
    ],
}

# Misleading surface: the title / size suggests one level, the substance another.
HARD: dict[str, list[tuple[str, str, list[str], tuple[int, int]]]] = {
    "skip": [
        ("Rename {mod} package across {svc}",
         "Mechanical rename via IDE refactor; no logic change.",
         ["{svc}/{mod}/__init__.py", "{svc}/{mod}/service.py", "{svc}/{mod}/views.py",
          "tests/{mod}/test_service.py"], (300, 1400)),
        ("Regenerate API client from spec", "Generated code only; spec unchanged.",
         ["clients/generated/{mod}_client.py"], (500, 2500)),
    ],
    "shallow": [
        ("Fix flaky {mod} test", "Freezes the clock in one test; production code untouched.",
         ["tests/{mod}/test_scheduler.py"], (3, 12)),
        ("Move {mod} constants to settings", "Same values, new location; behaviour unchanged.",
         ["{svc}/{mod}/settings.py", "{svc}/{mod}/service.py", "{svc}/config.py"], (40, 150)),
    ],
    "deep": [
        ("Fix typo in {mod} cache key", "Corrects `ordrs` to `orders` in the cache key prefix.",
         ["{svc}/{mod}/cache.py"], (1, 2)),
        ("Small cleanup in {mod} date handling", "Uses `datetime.now()` instead of `utcnow()`.",
         ["{svc}/{mod}/dates.py"], (1, 4)),
    ],
    "human": [
        ("Fix typo in auth check", "`is_admin` was misspelled in one branch; corrected.",
         ["{svc}/auth/permissions.py"], (1, 2)),
        ("Docs: update retention note", "Also changes the default retention from 30 to {n} days.",
         ["docs/privacy.md", "{svc}/settings.py"], (2, 8)),
        ("Bump payments SDK", "Major version bump of the payments SDK.",
         ["pyproject.toml", "poetry.lock"], (6, 60)),
    ],
}

# Share of each level in the pack. Skewed like real PR traffic: most changes are low risk.
MIX = {"skip": 0.35, "shallow": 0.30, "deep": 0.22, "human": 0.13}


def _case(i: int, level: str, hard: bool, rng: np.random.Generator) -> dict:
    title_t, body_t, files_t, (lo, hi) = (HARD if hard else EASY)[level][
        int(rng.integers(len((HARD if hard else EASY)[level])))
    ]
    # One fill context per case so {svc}/{mod} agree across title, body and files.
    ctx = {
        "svc": str(rng.choice(SERVICES)),
        "mod": str(rng.choice(MODULES)),
        "n": int(rng.integers(2, 9)),
    }
    title = title_t.format(**ctx)
    body = body_t.format(**ctx)
    files = [f.format(**ctx) for f in files_t]
    added = int(rng.integers(lo, hi + 1))
    removed = int(rng.integers(0, max(1, added // 2) + 1))
    author = str(rng.choice(PEOPLE))
    state = (
        f"PR title: {title}\n"
        f"Description: {body}\n"
        f"Files changed ({len(files)}): {', '.join(files)}\n"
        f"Lines: +{added} -{removed}\n"
        f"Author: {author}"
    )
    return {
        "id": f"pr-{i:04d}",
        "state": state,
        "question": QUESTION,
        "choices": list(LEVELS),
        "label": level,
        "metadata": {
            "domain": "code-review",
            "outcome": OUTCOME[level],
            "hard": hard,
            "files": files,
            "lines_added": added,
            "lines_removed": removed,
        },
    }


def build(n: int, seed: int, hard_rate: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    levels = rng.choice(LEVELS, size=n, p=[MIX[lv] for lv in LEVELS])
    return [
        _case(i, str(level), bool(rng.random() < hard_rate), rng)
        for i, level in enumerate(levels)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard-rate", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("examples/code-review"))
    args = parser.parse_args()
    if not 0.0 <= args.hard_rate <= 1.0:
        raise SystemExit("--hard-rate must be in [0, 1]")

    rows = build(args.n, args.seed, args.hard_rate)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {lv: sum(r["label"] == lv for r in rows) for lv in LEVELS}
    n_hard = sum(r["metadata"]["hard"] for r in rows)
    print(f"wrote {len(rows)} cases to {path}")
    print(f"labels {counts}  hard={n_hard}")


if __name__ == "__main__":
    main()
