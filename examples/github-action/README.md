# Thin gate: route pull requests with a certified profile

A GitHub Action that scores each PR with Jev Score, applies a certified
DecisionOps profile, and labels the PR with the route:

- `review:<level>` when the profile certifies the score (`auto`)
- `review:escalate` when it does not (below threshold, or in the audit sample)

DecisionOps decides the route only. What each label triggers is up to your
own workflows: an LLM reviewer on `review:deep`, a required human review on
`review:human` or `review:escalate`, nothing on `review:skip`. No LLM is called
from here.

**Status: not yet run on a real GitHub repository.** `pr_route.py` is covered
by tests with a fake Jev and was dry-run against live Jev (`--no-label`), but
the labels API calls, the workflow permissions and a real `pull_request`
payload have only been exercised with stubs. Expect small fixes on a first run.

## Files

- `pr_route.py` — PR event → `DecisionCase` → Jev Score → `decide` → label,
  step outputs, job summary, and one decision record.
- `decisionops-pr-route.yml` — the workflow to copy into your repository.
- `sample_event.json` — a PR event for a local dry run.

## Set up

1. Certify a profile on **your own** labelled PRs (see
   [../code-review/LABELLING.md](../code-review/LABELLING.md) and
   [../../docs/shadow-mode.md](../../docs/shadow-mode.md)). The profile in
   `examples/code-review/` was certified on synthetic PRs; it shows the
   mechanics, not a threshold for real traffic.
2. Commit `pr_route.py`, `profile.json` and `contract.json` under
   `.decisionops/` in the repository you route.
3. Add the `TYPESAFE_API_KEY` secret, and copy `decisionops-pr-route.yml` to
   `.github/workflows/`.
4. Pin the DecisionOps install to the tag or commit you certified with.

## Dry run locally

```bash
export TYPESAFE_API_KEY=...
uv run python examples/github-action/pr_route.py \
  --event examples/github-action/sample_event.json --no-label
```

`--no-label` never touches GitHub. The sample is the "typo fix in an auth
check" case; with the committed tolerance-1 profile it routes `auto → human`.

## Why it is shaped this way

- **The certificate holds for one input layout.** `case_from_pr` writes the
  same `state` lines (title, description, files, line counts, author) as the
  pack the profile was certified on. Change the layout, add fields, or switch
  the description length, and you must recertify.
- **The model is pinned.** The Jev call uses the profile's model id, not
  `jev-latest`. If Jev serves a different model, `decide` refuses
  (`ProfileMismatchError`) rather than route on an uncertified score.
- **The contract is checked.** A contract whose fingerprint differs from the
  profile's is refused for the same reason.
- **PR text is data.** It is read from the event file and sent to Jev as
  `state`; it is never interpolated into a shell command. The workflow checks
  out the base commit, so code from the PR never runs with secrets, and forks
  (which get no secrets) are skipped.
- **Every decision is kept.** `decision.jsonl` is uploaded as an artifact.
  Label the PRs where `audit` is `true` once their outcome has matured, and
  feed those to `decisionops recertify`. Escalated PRs stay out of
  recertification: they were picked by low confidence and are a biased sample.

## Outputs

The `route` step sets `action` (`auto` / `escalate`), `level`, `reason`,
`audit`, `confidence` and `label`, so later jobs can branch on them. The
workflow's `escalate` job is a placeholder for your own escalation path.
