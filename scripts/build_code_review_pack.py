#!/usr/bin/env python3
"""
Build the synthetic code-review showcase pack: one PR per case, labelled with
the review depth it actually needed (skip < shallow < deep < human).

Labels follow the rubric in examples/contracts/code-review-depth.json:

  skip     cannot change shipped behaviour: docs, comments, formatting,
           generated code, mechanical renames/moves, dev-only tooling
  shallow  touches shipped code or config but not production behaviour that
           matters: tests, log/message text, metrics, patch bumps of shipped
           dependencies, staging config, code behind a flag that is off
  deep     changes production logic: control flow, queries, caching,
           concurrency, retries, error handling, API contracts
  human    money, auth/permissions, personal data, irreversible migrations,
           secrets, production infrastructure

Each label comes from a simulated outcome (LABELLING.md): no_issue -> skip,
style_comment -> shallow, bug_in_deep_review -> deep, incident_or_revert ->
human.

Three kinds of case, with rates fixed before any model was run:

- plain: the surface matches the substance.
- hard (~20%): the title, size or path points the wrong way (a "typo fix" in an
  auth check, a 1,400-line mechanical rename). Reading the PR settles it.
- surprise (~5% of skip/shallow PRs): an ordinary-looking low-risk change that
  broke something after merge, so its label is `deep`. Nothing in the PR shows
  it. No decision model can see this risk; it is why a certificate bounds a
  miss rate instead of promising zero.

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
PEOPLE = ["dev-a", "dev-b", "dev-c", "dev-d", "dev-e", "dev-f", "dev-g", "dev-h"]

# (title, descriptions separated by " | ", files separated by ",", (min, max) lines)
# Placeholders: {svc} {mod} {n}. A description is picked at random per case.
T = tuple[str, str, str, tuple[int, int]]

PLAIN: dict[str, list[T]] = {
    "skip": [
        ("Fix typo in {mod} README",
         "Corrects spelling in the setup section. | 'recieve' -> 'receive'.",
         "docs/{mod}/README.md", (1, 4)),
        ("Update links in CONTRIBUTING", "The wiki moved; points links at the new pages.",
         "CONTRIBUTING.md", (1, 6)),
        ("Reword docstrings in {mod}",
         "Docstring wording only. | Clarifies parameter docs; no code.",
         "{svc}/{mod}/service.py", (2, 20)),
        ("Reformat {mod} with black", "Formatter run, whitespace only. | No manual edits.",
         "{svc}/{mod}/views.py,{svc}/{mod}/utils.py", (20, 300)),
        ("Add {mod} release notes", "CHANGELOG entry for the sprint.", "CHANGELOG.md", (3, 15)),
        ("Bump pre-commit hooks", "Dev-only lint hooks; nothing shipped.",
         ".pre-commit-config.yaml", (1, 6)),
        ("Rename helper in {mod}",
         "Renames `_fmt` to `_format_row`; callers updated, IDE refactor.",
         "{svc}/{mod}/utils.py,{svc}/{mod}/views.py", (6, 40)),
        ("Extract constant in {mod}",
         "Replaces a repeated literal with a named constant, same value.",
         "{svc}/{mod}/constants.py,{svc}/{mod}/service.py", (4, 20)),
        ("Sort imports in {svc}",
         "isort run across the service.", "{svc}/{mod}/service.py", (5, 60)),
        ("Fix broken badge in README", "CI badge URL changed.", "README.md", (1, 2)),
        ("Add architecture diagram for {mod}", "New docs page with a diagram.",
         "docs/{mod}/architecture.md", (20, 90)),
        ("Update editorconfig", "Tab width for YAML files.", ".editorconfig", (1, 3)),
    ],
    "shallow": [
        ("Add unit tests for {mod} parser",
         "Tests only; no production change. | Covers empty input.",
         "tests/{mod}/test_parser.py", (30, 160)),
        ("Clarify log message in {svc}", "Adds the request id to an info log line.",
         "{svc}/{mod}/handlers.py", (1, 6)),
        ("Patch-bump {mod} HTTP client",
         "Patch release of a shipped dependency; changelog has bug fixes only.",
         "pyproject.toml,poetry.lock", (4, 40)),
        ("Raise staging log level for {svc}", "Staging only; production config untouched.",
         "deploy/staging/{svc}.yaml", (1, 4)),
        ("Add metric for {mod} queue depth", "New gauge; no behaviour change.",
         "{svc}/{mod}/metrics.py,{svc}/{mod}/worker.py", (8, 30)),
        ("Improve {mod} validation error text", "Error message wording for the UI.",
         "{svc}/{mod}/errors.py", (2, 12)),
        ("Add {mod} fixture for integration tests", "Test data only.",
         "tests/fixtures/{mod}.json", (40, 200)),
        ("New {mod} endpoint behind flag", "Flag `{mod}_v2` defaults off in every environment.",
         "{svc}/{mod}/views_v2.py,{svc}/flags.py", (60, 180)),
        ("Update CI cache key",
         "CI config only; speeds up test jobs.", ".github/workflows/ci.yml", (2, 8)),
        ("Add type hints to {mod}",
         "Annotations only; mypy clean.", "{svc}/{mod}/service.py", (10, 80)),
        ("Translate {mod} emails to German", "New locale strings.",
         "{svc}/{mod}/locale/de.po", (30, 120)),
        ("Add debug endpoint to staging {svc}", "Registered only when ENV=staging.",
         "{svc}/debug.py", (15, 50)),
    ],
    "deep": [
        ("Add pagination to {mod} list endpoint", "Cursor parameter and page size limit.",
         "{svc}/{mod}/views.py,{svc}/{mod}/queries.py,tests/{mod}/test_views.py", (60, 220)),
        ("Cache {mod} lookups", "TTL cache around the lookup. | Memoises per request.",
         "{svc}/{mod}/cache.py,{svc}/{mod}/service.py", (30, 160)),
        ("Parallelise {mod} export", "Splits the export into {n} worker threads.",
         "worker/{mod}/export.py,worker/{mod}/pool.py", (80, 260)),
        ("Rewrite {mod} retry logic", "Backoff with jitter; max {n} attempts.",
         "{svc}/{mod}/retry.py,tests/{mod}/test_retry.py", (50, 180)),
        ("Change {mod} query to a join", "Replaces N+1 queries with one join.",
         "{svc}/{mod}/queries.py", (20, 90)),
        ("Handle timeouts in {svc} client", "Returns a partial result on timeout.",
         "{svc}/{mod}/client.py,{svc}/{mod}/errors.py", (30, 120)),
        ("Refactor {mod} service", "General cleanup and simplification. | Splits a long function.",
         "{svc}/{mod}/service.py", (80, 240)),
        ("Change default sort in {mod} API", "Sorts by updated_at instead of id.",
         "{svc}/{mod}/views.py", (2, 10)),
        ("Make {mod} sync idempotent", "Skips rows already processed.",
         "worker/{mod}/sync.py", (20, 90)),
        ("Add bulk update to {mod}", "New batch endpoint, {n}00 rows per call.",
         "{svc}/{mod}/views.py,{svc}/{mod}/queries.py", (60, 200)),
        ("Switch {mod} job to async", "asyncio instead of threads.",
         "worker/{mod}/job.py", (60, 220)),
        ("Remove {mod} fallback path", "Deletes the legacy fallback branch.",
         "{svc}/{mod}/service.py", (10, 60)),
    ],
    "human": [
        ("Change refund rounding", "Rounds refunds to the nearest cent instead of down.",
         "billing/refunds.py,billing/money.py", (8, 40)),
        ("Let support view {mod} admin", "Grants the support role read access to {mod} records.",
         "{svc}/auth/permissions.py,{svc}/{mod}/admin.py", (6, 30)),
        ("Drop legacy {mod} columns", "Migration removes {n} columns; no down migration.",
         "migrations/0{n}_drop_{mod}_legacy.py", (20, 80)),
        ("Load signing key from new vault path", "JWT signing key location change.",
         "{svc}/auth/keys.py,deploy/prod/{svc}.yaml", (10, 50)),
        ("Lower prod replicas for {svc}",
         "Min replicas from 6 to {n}.", "deploy/prod/{svc}.yaml", (2, 6)),
        ("Log {mod} payloads for debugging", "Logs request bodies, including email and phone.",
         "{svc}/{mod}/handlers.py", (2, 10)),
        ("Change tax calculation for EU", "VAT now applied before discounts.",
         "billing/tax.py", (20, 80)),
        ("Extend session lifetime", "Sessions now last {n}0 days instead of 7.",
         "{svc}/auth/session.py", (2, 8)),
        ("Export {mod} data to analytics vendor", "Sends user ids and emails to a third party.",
         "worker/{mod}/vendor_export.py", (40, 140)),
        ("Rotate prod database password", "New secret reference in prod config.",
         "deploy/prod/secrets.yaml", (2, 6)),
        ("Change payout schedule", "Merchant payouts weekly instead of daily.",
         "billing/payouts.py", (10, 50)),
        ("Open {svc} port to the internet", "Security group allows 0.0.0.0/0 on admin port.",
         "infra/prod/{svc}.tf", (2, 10)),
    ],
}

# The surface points the wrong way; reading the substance settles the level.
HARD: dict[str, list[T]] = {
    "skip": [
        ("Rename {mod} package across {svc}", "Mechanical IDE rename; no logic change.",
         "{svc}/{mod}/__init__.py,{svc}/{mod}/service.py,{svc}/{mod}/views.py,"
         "tests/{mod}/test_service.py", (300, 1400)),
        ("Regenerate API client", "Generated from an unchanged spec.",
         "clients/generated/{mod}_client.py", (500, 2500)),
        ("Fix comment in auth module", "Comment typo; no code touched.",
         "{svc}/auth/permissions.py", (1, 2)),
        ("Reformat billing package", "black run; whitespace only.",
         "billing/refunds.py,billing/money.py,billing/tax.py", (100, 600)),
    ],
    "shallow": [
        ("Fix flaky {mod} test",
         "Freezes the clock in one test.", "tests/{mod}/test_scheduler.py", (3, 12)),
        ("Add tests for refund rounding", "Tests only; billing code untouched.",
         "tests/billing/test_refunds.py", (40, 140)),
        ("Big logging cleanup in {svc}", "Rewords {n}0 log messages; no logic.",
         "{svc}/{mod}/handlers.py,{svc}/{mod}/service.py,{svc}/{mod}/worker.py", (200, 700)),
    ],
    "deep": [
        ("Fix typo in {mod} cache key", "`ordrs` -> `orders` in the cache key prefix.",
         "{svc}/{mod}/cache.py", (1, 2)),
        ("Small date cleanup in {mod}", "`datetime.now()` instead of `utcnow()`.",
         "{svc}/{mod}/dates.py", (1, 4)),
        ("Tidy {mod} config", "Also changes the default timeout from 30s to 3s.",
         "{svc}/{mod}/config.py", (2, 8)),
    ],
    "human": [
        ("Fix typo in auth check", "`is_admin` was misspelled in one branch; corrected.",
         "{svc}/auth/permissions.py", (1, 2)),
        ("Docs: update retention note", "Also changes the default retention from 30 to {n} days.",
         "docs/privacy.md,{svc}/settings.py", (2, 8)),
        ("Bump payments SDK",
         "Major version of the payments SDK.", "pyproject.toml,poetry.lock", (6, 60)),
        ("Clean up prod config", "Removes unused keys; also deletes the rate limit block.",
         "deploy/prod/{svc}.yaml", (4, 20)),
    ],
}

# Share of each level in the pack. Skewed like real PR traffic.
MIX = {"skip": 0.35, "shallow": 0.30, "deep": 0.22, "human": 0.13}


def _render(t: T, rng: np.random.Generator) -> tuple[str, str, list[str], int]:
    title_t, desc_t, files_t, (lo, hi) = t
    # One context per case so {svc}/{mod} agree across title, description and files.
    ctx = {
        "svc": str(rng.choice(SERVICES)),
        "mod": str(rng.choice(MODULES)),
        "n": int(rng.integers(2, 9)),
    }
    descs = desc_t.split(" | ")
    desc = descs[int(rng.integers(len(descs)))]
    files = [f.format(**ctx) for f in files_t.split(",")]
    return title_t.format(**ctx), desc.format(**ctx), files, int(rng.integers(lo, hi + 1))


def _case(i: int, level: str, kind: str, rng: np.random.Generator) -> dict:
    # A surprise looks like its surface level but is labelled deep.
    pool = HARD if kind == "hard" else PLAIN
    t = pool[level][int(rng.integers(len(pool[level])))]
    title, desc, files, added = _render(t, rng)
    label = "deep" if kind == "surprise" else level
    removed = int(rng.integers(0, max(1, added // 2) + 1))
    state = (
        f"PR title: {title}\n"
        f"Description: {desc}\n"
        f"Files changed ({len(files)}): {', '.join(files)}\n"
        f"Lines: +{added} -{removed}\n"
        f"Author: {rng.choice(PEOPLE)}"
    )
    return {
        "id": f"pr-{i:04d}",
        "state": state,
        "question": QUESTION,
        "choices": list(LEVELS),
        "label": label,
        "metadata": {
            "domain": "code-review",
            "outcome": OUTCOME[label],
            "kind": kind,
            "surface_level": level,
            "files": files,
            "lines_added": added,
            "lines_removed": removed,
        },
    }


def build(n: int, seed: int, hard_rate: float, surprise_rate: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    levels = rng.choice(LEVELS, size=n, p=[MIX[lv] for lv in LEVELS])
    rows = []
    for i, level in enumerate(levels):
        level = str(level)
        r = rng.random()
        if level in ("skip", "shallow") and r < surprise_rate:
            kind = "surprise"
        elif r < surprise_rate + hard_rate:
            kind = "hard"
        else:
            kind = "plain"
        rows.append(_case(i, level, kind, rng))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard-rate", type=float, default=0.2)
    parser.add_argument("--surprise-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("examples/code-review"))
    args = parser.parse_args()
    for name in ("hard_rate", "surprise_rate"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")

    rows = build(args.n, args.seed, args.hard_rate, args.surprise_rate)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {lv: sum(r["label"] == lv for r in rows) for lv in LEVELS}
    kinds = {
        k: sum(r["metadata"]["kind"] == k for r in rows)
        for k in ("plain", "hard", "surprise")
    }
    distinct = len({r["state"].splitlines()[1] for r in rows})
    print(f"wrote {len(rows)} cases to {path}")
    print(f"labels {counts}  kinds {kinds}  distinct descriptions {distinct}")


if __name__ == "__main__":
    main()
