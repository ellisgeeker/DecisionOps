# Code review showcase pack

Synthetic pull requests routed by review depth: `skip < shallow < deep < human`.
The decision model is hosted Jev, asked a **Score** question whose rubric is the
contract's ordered levels. DecisionOps certifies when that score can route a PR
automatically, and the calling application decides what each route does (merge,
ask an LLM for a deep review, page a human). No LLM is called from this library.

- `dataset.jsonl` — 600 PRs (title, description, files, line counts, author) with
  outcome-derived labels. About 17% are misleading on the surface, and about 4%
  are surprises: low-risk-looking changes that broke something after merge.
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

Rows are the true level, columns Jev's. Exact-level accuracy 94.2%.

| true \ Jev | skip | shallow | deep | human |
|---|--:|--:|--:|--:|
| skip | 183 | 0 | 0 | 0 |
| shallow | 8 | 171 | 0 | 0 |
| deep | 18 | 6 | 134 | 3 |
| human | 0 | 0 | 0 | 77 |

| Case kind | n | Jev exact | Misses, tolerance 1 | Misses, tolerance 0 |
|---|--:|--:|--:|--:|
| plain | 476 | 97.7% | 0 | 8 |
| hard (surface misleads) | 100 | 100% | 0 | 0 |
| surprise (risk not in the PR) | 24 | 0% | 18 | 24 |

Jev read the substance: every misleading PR ("typo fix" in an auth check,
1,400-line rename) got the right level. The misses that matter are the
surprises: ordinary docstring or import changes that broke something after
merge. Jev scored them `skip` with high confidence, which is the correct
reading of the text. No decision model can see that risk. A certificate can
only bound how often it lands on the automatic path.

The pack is synthetic and its cues are cleaner than real PRs, so the exact
numbers say little about production. What carries over is how to read them.

## Two contracts, same predictions (profiles at α = 5%)

| | `code-review-depth` (tolerance 1) | `code-review-depth-strict` (tolerance 0) |
|---|---|---|
| Miss means | under-rated by 2+ levels | under-rated by any level |
| Misses in 600 | 18 (3.0%), all surprises | 32 (5.3%) |
| Certified threshold | 0.308 (lowest confidence seen) | 0.570 |
| Held-out coverage / joint error | 100% / 2.9% | 59.3% / 1.9% |
| Mean over 30 split seeds | 97.2% / 2.9% | 78.2% / 3.9% |
| `decide` on all 600 | 573 auto, 27 audit | 342 auto, 240 below threshold, 18 audit |

**Tolerance 1: full automation, with the risk stated.** The unseen-risk rate
(3%) is already under α = 5%, so CRC needs no threshold and every PR may route
automatically. The certificate says the miss rate is ≤ 5% in expectation, and
it is: 2.9% on held-out PRs. If surprises rose above α, the same certificate
would start holding traffic back, and the audit sample is what notices. Under
`--confidence-level 0.95` (a per-certificate guarantee) the threshold rises to
0.551 and coverage falls to 64%.

**Tolerance 0: what tightening costs.** Calling a `shallow` PR `skip` now
counts too, and the total miss rate (5.3%) is above α, so the certificate has to
withhold traffic. The misses are confident, so a confidence threshold catches
them inefficiently: 240 PRs go to the expensive path to catch 16 misses, and 15
surprises still run automatically. The joint error stays under α (1.9% held
out, 3.9% over 30 seeds); the price is automation, 57% instead of 96%.

Which tolerance is right is a product decision. Tolerance 1 trusts that a
one-level under-rate is harmless; tolerance 0 pays for that doubt in review
time. Temperature here is fit on 36 held-out PRs (T ≈ 6.5): most confidences
sit near 1.0 and the few confident misses pull T up, flattening scores before
CRC sees them. Thresholds are on that scale.

## Reproduce

```bash
D=examples/code-review
for v in default strict; do
  c=examples/contracts/code-review-depth.json
  [ $v = strict ] && c=examples/contracts/code-review-depth-strict.json
  uv run decisionops certify $D/providers/jev_dataset.jsonl -p $D/providers/jev.jsonl \
    --contract $c --alphas 0.05,0.10 --profile $D/$v/profile.json --profile-alpha 0.05 \
    --audit-rate 0.05 -o $D/$v/certify
  uv run decisionops decide $D/$v/profile.json -p $D/providers/jev.jsonl \
    -o $D/$v/decisions.jsonl
done
```

`default/` and `strict/` hold the committed reports, profiles and decisions.
What a route *does* (merge, call an LLM reviewer, page a human) belongs to the
calling application: DecisionOps only returns `auto` with the chosen level, or
`escalate`.
