# Labelling guide: how much review did this PR need?

A label is the review depth a pull request **turned out to need**, decided after
the fact from what happened to it. It is not a reviewer's first guess, and it is
not what the model predicted. A certificate is only as good as these labels.

## Levels (low → high)

| Level | Rule | Typical changes |
|---|---|---|
| `skip` | Cannot change shipped behaviour | Docs, comments, formatting, generated code from an unchanged spec, mechanical renames or moves with identical values, dev-only tooling |
| `shallow` | Touches shipped code or config, but not production behaviour that matters | Tests, log or message text, metrics, type hints, patch bumps of shipped dependencies, staging or CI config, code behind a flag that is off |
| `deep` | Changes production logic | Control flow, defaults, queries, caching, concurrency, retries, error handling, API behaviour |
| `human` | A responsible human must sign off | Money, auth / permissions, sessions, personal data, irreversible migrations, secrets, production infrastructure or security rules |

Judge the substance, not the surface. A one-line "typo fix" inside a permission
check is `human`. A 1,400-line IDE rename is `skip`. The contract sends these
same rules to the model as its rubric; if the rubric and the labels disagree,
the certificate measures the disagreement, not the model.

## Where the label comes from

Use the strongest evidence available, in this order:

1. **Incident or revert** traced to the PR → `human` if it touched a `human`
   area, otherwise at least `deep`.
2. **Defect found in a deep review** (LLM or specialist) and confirmed by the
   author → `deep`.
3. **Policy**: changes in a `human` area are `human` even when nothing went
   wrong, because the sign-off itself is the requirement.
4. **Quick review left only cosmetic comments**, nothing changed in behaviour →
   `shallow`.
5. **Merged, nothing found, never reverted** within the maturity window → the
   lowest level that matches the change type above (`skip` or `shallow`).

Do not label a case before its maturity window has passed (for example 14 days
after merge). A revert that arrives later changes the label; that is why
`recertify` only reads outcomes whose maturity time is set.

## What a miss is

The contract (`examples/contracts/code-review-depth.json`) uses ordinal risk
with `under_rate_tolerance: 1`:

- Under-rating by **two or more levels** is a miss: `skip` for a `deep` PR,
  `shallow` for a `human` PR.
- Under-rating by one level (`deep` for a `human` PR) is tolerated.
- Over-rating never counts as risk. It costs review time, not safety.

If one-level under-rating is not acceptable for your team, set the tolerance to
0 and recertify.

## Which cases to label after go-live

Label the **random audit sample** that `decide --audit-rate` marks
(`audit: true`), plus everything that was escalated anyway. Only the audit rows
feed `recertify`: escalated cases were picked by low confidence, so they are a
biased sample. See [docs/shadow-mode.md](../../docs/shadow-mode.md).

## In this pack

The 600 PRs here are synthetic. Each row's `metadata.outcome` records the
simulated evidence (`no_issue`, `style_comment`, `bug_in_deep_review`,
`incident_or_revert`). `metadata.kind` is one of:

- `plain` — the surface matches the substance;
- `hard` (~17%) — the title, size or path points the wrong way;
- `surprise` (~4%) — an ordinary-looking `skip` / `shallow` change that broke
  something after merge, so it is labelled `deep`. `metadata.surface_level` is
  what it looked like. Nothing in the PR reveals it.
