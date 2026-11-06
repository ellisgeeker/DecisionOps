#!/usr/bin/env python3
"""
Route one pull request with a certified DecisionOps profile.

Thin gate for GitHub Actions. It does four things and nothing else:

1. Build a DecisionCase from the PR (title, description, changed files, line
   counts, author) in the same text layout the certificate was calibrated on.
2. Ask hosted Jev a Score question, pinned to the model the profile certifies.
3. Run `decisionops.decide` with the profile: `auto` + level, or `escalate`.
4. Report the result: step outputs, a job summary, a `review:<level>` or
   `review:escalate` label, and one decision record (JSON line) for the audit
   loop.

What a route *does* (merge, call an LLM reviewer, request a human) is the
calling workflow's job. This script never calls an LLM and never merges.

The PR text is untrusted. It is read from the event file and sent to Jev as
data; it is never interpolated into a shell command.

Local dry run without GitHub (no labels written):

  TYPESAFE_API_KEY=... python examples/github-action/pr_route.py \\
    --event examples/github-action/sample_event.json --no-label
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from decisionops.contract import DecisionContract, load_contract
from decisionops.decide import (
    CertificationProfile,
    Decision,
    decide,
    load_profile,
    require_single_profile,
)
from decisionops.schemas import DecisionCase, DecisionPrediction

QUESTION = "How much review does this pull request need before merging?"
LABEL_PREFIX = "review:"
MAX_DESCRIPTION = 600
MAX_FILES = 40
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


# --------------------------------------------------------------- PR -> case


def _one_line(text: str | None, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def case_from_pr(pr: dict[str, Any], files: list[str], choices: list[str]) -> DecisionCase:
    """
    The same `state` layout as examples/code-review. A certificate holds only
    for the input distribution it was calibrated on, so keep this in step with
    the pack you certified.
    """
    shown = files[:MAX_FILES]
    more = f" (+{len(files) - MAX_FILES} more)" if len(files) > MAX_FILES else ""
    state = (
        f"PR title: {_one_line(pr.get('title'), 200)}\n"
        f"Description: {_one_line(pr.get('body'), MAX_DESCRIPTION) or '(none)'}\n"
        f"Files changed ({len(files)}): {', '.join(shown)}{more}\n"
        f"Lines: +{int(pr.get('additions') or 0)} -{int(pr.get('deletions') or 0)}\n"
        f"Author: {(pr.get('user') or {}).get('login', 'unknown')}"
    )
    # The label is unknown at decision time; DecisionCase needs one, and it is
    # never sent to the model (adapters send only state / question / choices).
    return DecisionCase(
        id=f"pr-{pr['number']}",
        state=state,
        question=QUESTION,
        choices=list(choices),
        label=choices[0],
        metadata={"head_sha": (pr.get("head") or {}).get("sha")},
    )


# --------------------------------------------------------------- GitHub API


class GitHub:
    """Minimal REST client: list PR files, set the route label."""

    def __init__(self, repo: str, token: str | None) -> None:
        self.repo = repo
        self.token = token

    def _call(self, method: str, path: str, body: Any | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{API}{path}", data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def pr_files(self, number: int) -> list[str]:
        files: list[str] = []
        page = 1
        while True:
            batch = self._call(
                "GET", f"/repos/{self.repo}/pulls/{number}/files?per_page=100&page={page}"
            )
            files += [f["filename"] for f in batch]
            if len(batch) < 100 or page >= 30:  # API caps the list at 3,000 files
                return files
            page += 1

    def set_route_label(self, number: int, label: str, current: list[str]) -> None:
        """Replace any earlier `review:*` label, so a re-run never leaves two."""
        for old in current:
            if old.startswith(LABEL_PREFIX) and old != label:
                try:
                    self._call(
                        "DELETE",
                        f"/repos/{self.repo}/issues/{number}/labels/"
                        + urllib.request.quote(old, safe=""),
                    )
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
        if label not in current:
            self._call("POST", f"/repos/{self.repo}/issues/{number}/labels", {"labels": [label]})


# --------------------------------------------------------------- routing


def route_label(decision: Decision) -> str:
    target = decision.choice if decision.action == "auto" else "escalate"
    return f"{LABEL_PREFIX}{target}"


def route(
    case: DecisionCase,
    profile: CertificationProfile,
    contract: DecisionContract,
    provider: Any,
) -> tuple[DecisionPrediction, Decision]:
    """Score one case and apply the certified rule. `provider` has `.predict(case)`."""
    profile.check_contract(contract)
    prediction = provider.predict(case)
    return prediction, decide(prediction, profile)


def jev_provider(profile: CertificationProfile, contract: DecisionContract):
    from decisionops.adapters.jev import JevProvider

    if not contract.is_ordinal:
        raise SystemExit("this gate asks Jev Score; the contract needs choice_order")
    # Pin the certified model: `jev-latest` moving on would void the certificate.
    return JevProvider(
        model=profile.model or "jev-latest",
        question_type="score",
        choice_order=contract.choice_order,
        descriptions=contract.descriptions,
        contract_instructions=contract.instructions,
    )


# --------------------------------------------------------------- reporting


def summary_markdown(case: DecisionCase, pred: DecisionPrediction, d: Decision) -> str:
    probs = " · ".join(f"{k} {v:.2f}" for k, v in pred.probabilities.items())
    verdict = (
        f"**auto → `{d.choice}`** (certified)"
        if d.action == "auto"
        else f"**escalate** ({d.reason.replace('_', ' ')})"
    )
    return "\n".join(
        [
            f"### DecisionOps route for #{case.id.removeprefix('pr-')}",
            "",
            verdict,
            "",
            "| | |",
            "|---|---|",
            f"| Jev level | `{d.choice}` |",
            f"| Confidence (calibrated) | {d.confidence:.3f} |",
            f"| Certified threshold | {d.threshold:.3f} at α = {d.alpha:g} |",
            f"| Audit sample | {'yes, label this PR' if d.audit else 'no'} |",
            f"| Model | {pred.model} |",
            f"| Profile | `{d.profile}` |",
            "",
            f"Jev level probabilities (raw, before the profile's temperature): {probs}",
        ]
    )


def _write_outputs(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for k, v in values.items():
            f.write(f"{k}={v}\n")


def _append(path: str | None, text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# --------------------------------------------------------------- main


def main(argv: list[str] | None = None, *, provider: Any | None = None) -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument(
        "--profile", type=Path, default=repo_root / "examples/code-review/default/profile.json"
    )
    parser.add_argument(
        "--contract", type=Path, default=repo_root / "examples/contracts/code-review-depth.json"
    )
    parser.add_argument("--decisions-out", type=Path, default=Path("decision.jsonl"))
    parser.add_argument("--no-label", action="store_true", help="dry run: do not touch the PR")
    args = parser.parse_args(argv)

    if not args.event:
        raise SystemExit("--event (or GITHUB_EVENT_PATH) is required")
    event = json.loads(Path(args.event).read_text(encoding="utf-8"))
    pr = event.get("pull_request")
    if not pr:
        raise SystemExit("event has no pull_request; run on pull_request events")

    profile = require_single_profile(load_profile(args.profile), what="pr_route")
    contract = load_contract(args.contract)
    choices = contract.choice_order or contract.choices

    repo = (event.get("repository") or {}).get("full_name") or os.environ.get("GITHUB_REPOSITORY")
    gh = GitHub(repo, os.environ.get("GITHUB_TOKEN")) if repo else None
    files = event.get("files")  # sample events may inline the file list
    if files is None:
        if gh is None:
            raise SystemExit("no repository in event or GITHUB_REPOSITORY; cannot list files")
        files = gh.pr_files(int(pr["number"]))

    case = case_from_pr(pr, files, choices)
    pred, decision = route(case, profile, contract, provider or jev_provider(profile, contract))
    label = route_label(decision)

    record = decision.model_dump(mode="json") | {
        "head_sha": case.metadata.get("head_sha"),
        "model": pred.model,
    }
    args.decisions_out.write_text(json.dumps(record) + "\n", encoding="utf-8")

    md = summary_markdown(case, pred, decision)
    print(md)
    _append(os.environ.get("GITHUB_STEP_SUMMARY"), md)
    _write_outputs(
        os.environ.get("GITHUB_OUTPUT"),
        {
            "action": decision.action,
            "level": decision.choice,
            "reason": decision.reason,
            "audit": str(decision.audit).lower(),
            "confidence": f"{decision.confidence:.4f}",
            "label": label,
        },
    )

    if not args.no_label:
        if gh is None:
            raise SystemExit("cannot label without a repository; pass --no-label for a dry run")
        current = [lab["name"] for lab in pr.get("labels") or []]
        gh.set_route_label(int(pr["number"]), label, current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
