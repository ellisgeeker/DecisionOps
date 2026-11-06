# Code review showcase pack

Synthetic pull requests routed by review depth: `skip < shallow < deep < human`.
The decision model is hosted Jev, asked a **Score** question whose rubric is the
contract's ordered levels. DecisionOps certifies when that score can route a PR
automatically, and the calling application decides what each route does (merge,
ask an LLM for a deep review, page a human). No LLM is called from this library.

- `dataset.jsonl` — 600 PRs (title, description, files, line counts, author) with
  outcome-derived labels. About 20% are deliberately misleading on the surface.
- `LABELLING.md` — how labels are defined and collected.
- Contract: [`examples/contracts/code-review-depth.json`](../contracts/code-review-depth.json)
  (ordinal, under-rate tolerance 1, costs per level).

This is not production traffic. It shows the full path on data you can read.

## Rebuild the data

```bash
uv run python scripts/build_code_review_pack.py --n 600 --seed 7 --out examples/code-review
```

Deterministic for a given seed.

## Score it with Jev (600 paid calls)

```bash
uv sync --extra jev
export TYPESAFE_API_KEY=...
uv run python scripts/export_jev_predictions.py --pack code-review --limit 0 \
  --contract examples/contracts/code-review-depth.json
```

The contract is ordinal, so the export asks Jev Score. It writes
`providers/jev.jsonl`, `providers/jev_dataset.jsonl` and `providers/jev_stats.json`.
Those files are committed, so everything below runs offline.

## What Jev did (jev-1.14.0, 600 PRs)

Exact-level accuracy 82.3%. Every error is a skip/shallow swap; `deep` and
`human` PRs were all scored correctly, including the misleading ones
("typo fix" in an auth check). Rows are the true level, columns Jev's.

| true \ Jev | skip | shallow | deep | human |
|---|--:|--:|--:|--:|
| skip | 164 | 35 | 0 | 0 |
| shallow | 71 | 116 | 0 | 0 |
| deep | 0 | 0 | 137 | 0 |
| human | 0 | 0 | 0 | 77 |

The pack is synthetic and its `deep` / `human` cues are clear, so this says
little about real PRs. It is enough to show the two ways a certificate reads.

## Two contracts, same predictions

| | `code-review-depth` (tolerance 1) | `code-review-depth-strict` (tolerance 0) |
|---|---|---|
| Miss means | under-rated by 2+ levels | under-rated by any level |
| Ordinal misses in 600 | 0 | 71 (all shallow PRs scored skip) |
| Certified threshold, α = 10% | 0.366 (lowest confidence seen) | 0.504 |
| Certified threshold, α = 5% | 0.366 | 0.566 |
| Test coverage / joint error, α = 10% | 100% / 0.0% | 94.8% / 8.1% |
| Test coverage / joint error, α = 5% | 100% / 0.0% | 88.3% / 3.8% |
| `decide` on all 600, α = 10% profile | 573 auto, 27 audit | 546 auto, 29 below threshold, 25 audit |

**Tolerance 1: everything may run automatically.** Jev never under-rated a PR
by two levels, so no threshold is needed to keep that risk under α. The
certificate is the evidence for full automation; the 5% audit sample is what
keeps checking it after go-live.

**Tolerance 0: the certificate holds traffic back.** Calling a `shallow` PR
`skip` now counts, and those calls come with lower confidence, so CRC raises
the threshold. At α = 10% the 29 PRs below it go to the expensive path, and 23
of them were real misses ("Move constants to settings" scored `skip`). The
other 47 misses had high confidence and still run automatically: the
certificate does not promise zero misses, it promises the miss rate stays under
α. On all 600 that is 47 / 600 = 7.8%. On held-out PRs it is 8.1% for this split
and 7.9% averaged over 30 split seeds, under α = 10% as an in-expectation
guarantee promises. For a tighter bound use α = 5% or `--confidence-level 0.95`.

Which tolerance is right is a product decision: whether a skipped quick look on
a `shallow` PR matters to you.

## Reproduce

```bash
D=examples/code-review
for v in default strict; do
  c=examples/contracts/code-review-depth.json
  [ $v = strict ] && c=examples/contracts/code-review-depth-strict.json
  uv run decisionops certify $D/providers/jev_dataset.jsonl -p $D/providers/jev.jsonl \
    --contract $c --alphas 0.05,0.10 --profile $D/$v/profile.json --profile-alpha 0.10 \
    --audit-rate 0.05 -o $D/$v/certify
  uv run decisionops decide $D/$v/profile.json -p $D/providers/jev.jsonl \
    -o $D/$v/decisions.jsonl
done
```

`default/` and `strict/` hold the committed reports, profiles and decisions.
What a route *does* (merge, call an LLM reviewer, page a human) belongs to the
calling application: DecisionOps only returns `auto` with the chosen level, or
`escalate`.
