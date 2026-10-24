# Findings

**Last updated:** 2026-09-27

Measured results, with the protocol and limits of each. Numbers here came from
runs on public benchmarks; nothing is synthetic unless labelled so.

---

## Setup

| | |
|---|---|
| Datasets | BANKING77 (`mteb/banking77`, 3076 held-out rows, 77 classes), CLINC OOS `plus` (`clinc/clinc_oos`, 8600 held-out rows, 151 classes) |
| Models | Laya `convaiinnovations/laya` (zero-shot) + TypeSafe Jev `jev-1.13.0` (hosted, zero-shot) + 4 sklearn heads |
| Protocol | Classifier fit on the HF **train** split; only **held-out** rows exported. DecisionOps then splits that pool into calibration / test (seed 42, `cal_ratio=0.3`). Temperature and thresholds come from calibration; all reported metrics come from test. |
| Hardware | Apple Silicon, MPS. Laya ≈98ms/case. |

---

## 1. A shipped post-hoc calibration made calibration worse

Laya ships per-option-count temperatures in `rl_agent_config.json`. For a 77-way
question it selects the `choice:11+` bucket, `T = 0.1006`. Temperatures below 1
**sharpen** a distribution.

BANKING77, n=3076, test split n=2153:

| Condition | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|
| Laya, bundled `T=0.10` | 0.375 | **0.603** | 1.221 | 15.04 |
| Laya, no bundled T (`T=1.0`) | 0.375 | 0.362 | 0.915 | 3.55 |
| Laya + temperature fit on held-out (`T≈6.9`) | 0.375 | **0.056** | 0.839 | 3.18 |

Turning the bundled temperature *off* nearly halved ECE. Fitting a temperature on
held-out data cut it by 91%.

This is not a claim that Laya's temperatures are wrong in general — other buckets
point the right way (`choice:3-5` uses `T=1.76`, which softens). It is a claim
that a shipped calibration table cannot be assumed correct for your option count
and your data distribution.

**Reproduce:**

```bash
uv run python scripts/export_laya_predictions.py --pack banking77 --limit 0
uv run python scripts/export_laya_predictions.py --pack banking77 --limit 0 \
  --no-builtin-temperature
uv run decisionops gate examples/banking77/providers/laya_dataset.jsonl \
  --predictions examples/banking77/providers/laya.jsonl \
  --output examples/banking77/gate-laya
```

---

## 2. Confidence can carry almost no information

With the bundled `T=0.10`, Laya's mean max-probability was **0.978** while
accuracy was **0.375**.

Sharpening pushed so much mass to the ceiling that a threshold of **1.000** still
accepted 41.2% of test traffic — and **44.7% of accepted cases were wrong**.

| Threshold | Test coverage | Selective risk |
|---|---|---|
| 1.000 | 41.2% | **0.447** |

A rule like `if confidence > 0.9: auto_execute()` would have automated something
close to a coin flip while reporting 99%+ confidence. The threshold mechanism
does not fail loudly here; it fails silently.

---

## 3. Good calibration does not mean safe automation

These are independent axes, and a calibration library only measures the first.

BANKING77, 40 seeds reshuffling the **same** 3076-row held-out pool (this
estimates the pool, not a population). Certified coverage is test-set coverage
at the CRC threshold, among seeds that found a feasible λ̂. `n_crc_cal=738`.

| Model | Acc | ECE (cal) | @1% feasible | @1% cov mean [p10, p90] | @10% feasible | @10% cov mean |
|---|---|---|---|---|---|---|
| Laya | 0.375 | **0.056** | **6/40** | 1.7% [0.4, 1.5] | 21/40 | 8.8% |
| word_lr | 0.850 | 0.034 | 40/40 | 58.2% [50.6, 65.2] | 40/40 | 93.9% |
| char_lr | 0.889 | 0.021 | 40/40 | **68.5%** [59.3, 75.7] | 40/40 | 99.1% |
| svm_sig | 0.877 | 0.017 | 40/40 | 66.0% [58.5, 71.5] | 40/40 | 98.1% |
| cnb | 0.689 | — | 40/40 | 26.5% [23.5, 28.7] | 40/40 | 59.7% |

Laya's calibrated ECE (0.056) is close to word_lr's (0.034). Its certified
coverage at 1% is not. 34 of 40 seeds found **no** threshold that can hold a 1%
joint-error budget — 37.5% accuracy leaves almost no separation between right
and wrong at the top of the ranking. When a threshold does exist, it covers
about 2% of traffic.

Loosening the budget does not make Laya usable. At 10%, 21/40 seeds find a
threshold, but mean coverage among those is still 8.8%. Seed 42 — the one
`certify` prints by default — happens to be in the infeasible half at both 1%
and 10%, which is why a single run looks like a hard zero.

A previous version of this table quoted char_lr @1% as "~75%" and Laya as
"0% (infeasible)". 75% is this pool's p90, not its typical value; Laya's 0% is
the typical *outcome* (infeasible) but not a property of every split. CRC still
does the useful thing: it refuses a certificate it cannot back, rather than
returning a threshold that would fail in production.

**Reproduce:**

```bash
uv run python scripts/resample_crc.py --seeds 40
uv run decisionops certify examples/banking77/providers/laya_dataset.jsonl \
  -p examples/banking77/providers/laya.jsonl --alphas 0.01,0.10
```

---

## 4. Empirically tuned thresholds rebound on held-out data

Scanning a calibration split for the lowest threshold whose empirical selective
risk is ≤ α means selecting for a downward fluctuation. That selection does not
transfer.

BANKING77 at a 1% target, thresholds chosen on calibration (n=923), measured on
test (n=2153):

| Model | Cal coverage | Test coverage | Test selective risk | Held? |
|---|---|---|---|---|
| word_lr | 64.6% | 61.3% | 0.029 | no |
| char_lr | 78.4% | 74.9% | **0.013** | no |
| svm_sig | 48.2% | 45.5% | 0.012 | no |
| laya | 5.6% | 5.1% | 0.012 | no |
| cnb | 3.0% | 2.8% | 0.000 | yes |

All five missed the 1% target on test (the one that "held" did so by accepting
almost nothing). Coverage also dropped in every case — the classic winner's-curse
signature.

`compare` marks these with `!` and prints "treat as not certified" rather than
reporting the target as met. **The `!` marks are the feature.** A tool that
reported only accuracy and ECE would have said char_lr looks great (88.9%, ECE
0.021) without mentioning that its 1% promise becomes 1.30% on new data.

---

## 5. Conformal Risk Control fixes the rebound, and was validated

CRC adds a `B/(n+1)` penalty before taking the smallest feasible threshold:

```
λ̂ = inf { λ : (n/(n+1))·R̂ₙ(λ) + B/(n+1) ≤ α }
```

giving `E[L_{n+1}(λ̂)] ≤ α` in finite samples under exchangeability.

### Validation against a closed-form optimum

Checking a bound by eyeballing empirical error means is too noisy to be
conclusive. Instead, `scripts/diagnose_crc.py` (and the regression test
`tests/test_crc_analytic.py`) builds a synthetic pool where the true answer is
known: confidence ~ U(0.55, 0.999), perfectly calibrated, so

```
true_joint_error(λ) = ((1-λ)² - 0.001²) / (2 · 0.449)
λ*(α)               = 1 - sqrt(2 · (0.449α + 5e-7))
```

A correct rule puts `λ̂` **at or above** `λ*`. 300 repetitions, n_cal=600:

| α | Rule | λ* | mean λ̂ | z | Implied true risk |
|---|---|---|---|---|---|
| 1% | `>=` | 0.90523 | 0.90922 | **+3.60** | 0.00918 |
| 1% | `>` | 0.90523 | 0.90837 | +2.84 | 0.00935 |
| 5% | `>=` | 0.78810 | 0.78983 | +1.59 | 0.04919 |
| 5% | `>` | 0.78810 | 0.78906 | +0.88 | 0.04955 |
| 10% | `>=` | 0.70033 | 0.70153 | +1.18 | 0.09920 |
| 10% | `>` | 0.70033 | 0.70079 | +0.45 | 0.09969 |

Conservative in all six configurations. Implied true risk sits below α
throughout. `>=` is the more conservative of the two acceptance conventions.

**Reproduce:** `uv run python scripts/diagnose_crc.py` (300 reps, both
acceptance conventions) or `uv run pytest tests/test_crc_analytic.py` (60 reps,
`>=` only, n_cal=150 so the `B/(n+1)` term is load-bearing).

### Minimum calibration size

The `B/(n+1)` term cannot be removed, so `α ≥ B/(n+1)` is necessary:

| Target α | Minimum calibration samples |
|---|---|
| 1% | **99** |
| 5% | 19 |
| 10% | 9 |

`min_calibration_samples()` returns this. Below the threshold, CRC reports
infeasible rather than guessing — which is the honest answer, and also explains
why small-sample empirical thresholds can appear to hold 1% when there is no
statistical basis for the claim.

### What is certified

The **joint automated error rate**: the share of *all* traffic that is
auto-accepted *and* wrong. **Not** conditional selective risk (error rate *among*
accepted cases), which is a ratio of two random quantities and breaks the
monotone-bounded-loss requirement. See [status.md](status.md#known-gaps).

### Using it

`decisionops certify` exposes this path. On BANKING77 / `char_lr`, certified
thresholds at `n_crc_cal=738` (single split, seed 42):

| α | Certified λ̂ | CRC bound | Coverage (cal) | Coverage (test) | Joint err (test) |
|---|---|---|---|---|---|
| 1% | 0.8564 | 0.0095 | 80.1% | 75.5% | 0.0149 |
| 5% | 0.5457 | 0.0487 | 92.4% | 91.5% | 0.0627 |
| 10% | 0.1184 | 0.0934 | 100.0% | 100.0% | 0.1036 |

All three test joint-error figures land just above α on *this* split, which is
the expected behaviour described above, not a failure — the smallest feasible λ̂
puts true risk just under α, so a single draw straddles it. Coverage barely moved
between calibration and test (80.1% → 75.5% at 1%), unlike the empirical
thresholds in §4. Across 40 seeds the typical char_lr @1% coverage is 68.5%,
with 75.5% sitting at this pool's p90 — another reason not to quote a single
split as the number.

```bash
uv run decisionops certify examples/banking77/dataset.jsonl \
  -p examples/banking77/providers/tfidf_char_lr.jsonl --alphas 0.01,0.05,0.10
```

---

## 6. Temperature scaling works, consistently

Four models, two datasets, one scalar parameter fit on calibration and reported
on test:

| Model | Dataset | ECE raw | ECE calibrated | Drop |
|---|---|---|---|---|
| laya | BANKING77 | 0.603 | 0.056 | 91% |
| word_lr | BANKING77 | 0.430 | 0.034 | 92% |
| char_lr | BANKING77 | 0.261 | 0.021 | 92% |
| svm_sig | BANKING77 | 0.124 | 0.017 | 86% |
| word_lr | CLINC OOS | 0.431 | 0.037 | 91% |
| char_lr | CLINC OOS | 0.285 | 0.028 | 90% |
| svm_sig | CLINC OOS | 0.105 | 0.013 | 88% |

86–92% reduction across the board. Accuracy is unchanged by construction —
temperature scaling is monotone, so the argmax does not move.

---

## 7. Which model is "best" depends on the risk budget

`decisionops compare` now ranks on CRC-certified coverage. BANKING77, seed 42,
`n_crc_cal=738` (single split — see §3 for split-to-split range):

| Model | Acc | ECE cal | @1% certified cov | @5% | @10% |
|---|---|---|---|---|---|
| svm_sig | 0.877 | 0.017 | **76.6%** | 90.5% | 99.9% |
| char_lr | 0.896 | 0.021 | 75.5% | **91.5%** | **100%** |
| word_lr | 0.850 | 0.030 | 61.6% | 85.8% | 98.9% |
| cnb | 0.683 | 0.131 | 27.7% | 45.1% | 62.3% |
| laya | 0.375 | 0.056 | infeasible | — | infeasible |

At 1% the winner is svm_sig, not the accuracy leader. At 5% and 10%, char_lr
wins. Laya is excluded: no certificate. A single accuracy leaderboard would have
picked char_lr and been wrong for the strictest requirement.

The older CLINC table below is **empirical** (pre-CRC ranking). Kept because the
qualitative pattern is the same, and because CLINC has not been re-run through
the certified `compare` path.

CLINC OOS, thresholds from calibration, coverage on test:

| Model | @1% | @5% | @10% |
|---|---|---|---|
| word_lr | 0.0% | 70.7% | 84.4% |
| char_lr | 0.0% | **78.9%** | **90.1%** |
| svm_sig | **40.5%** | 74.8% | 87.5% |
| cnb | 0.5% | 0.5% | 45.0% |

```bash
uv run decisionops compare examples/banking77/dataset.jsonl \
  -p char_lr=examples/banking77/providers/tfidf_char_lr.jsonl \
  -p svm_sig=examples/banking77/providers/tfidf_svm_sigmoid.jsonl \
  -p word_lr=examples/banking77/providers/tfidf_word_lr.jsonl \
  -p cnb=examples/banking77/providers/tfidf_cnb.jsonl \
  -p laya=examples/banking77/providers/laya.jsonl
```

---

## 8. Jev sits between Laya and a trained head — on the same 500 cases

Toy routing (20 cases, accuracy 1.0) said nothing about calibration. A live
BANKING77 export does.

500 held-out rows, same ids for every model (`scripts/export_jev_predictions.py
--pack banking77 --limit 500 --seed 42`, then the matching rows from the existing
Laya / sklearn JSONL). Jev is `jev-1.13.0` via `jev-latest`, zero-shot, 77
options per call. Sklearn heads were fit on the HF **train** split; Jev and Laya
were not.

Pool accuracy on those 500: Jev **0.794**, mean max-prob 0.878, 231 ms/case
(~1023 in / 828 out tokens). 103 wrong; 18 of those had confidence ≥ 0.9.
Most common confusion: `get_physical_card` → `change_pin` (10×).

Shared DecisionOps split on that pool (seed 42, `n_cal=150`, `n_test=350`,
`n_crc_cal=120`). 1% is omitted: 120 calibration samples is only just above the
CRC floor of 99, so a 1% certificate would be almost all penalty.

| Model | Acc (test) | ECE raw | ECE cal | @5% CRC cov | @10% CRC cov |
|---|---|---|---|---|---|
| svm_sig | 0.903 | 0.148 | 0.034 | **87.7%** | 95.1% |
| char_lr | **0.911** | 0.257 | 0.034 | 86.9% | **98.3%** |
| word_lr | 0.886 | 0.459 | **0.019** | 85.4% | 96.3% |
| **jev** | **0.777** | **0.094** | 0.092 | **67.1%** | **79.4%** |
| laya_rawT | 0.374 | 0.374 | 0.123 | 6.0% | 14.0% |
| laya | 0.374 | 0.607 | 0.052 | infeasible | infeasible |

Three things this table is for:

1. **Jev is already close to calibrated.** Temperature moves ECE 0.094 → 0.092
   (`T=3.24`). NLL still drops (2.32 → 1.12), so the scalar is doing work, but
   there is no Laya-style "shipped T made it worse" story. Laya on the same 350
   still needs `T≈6.6` to cut ECE from 0.61 to 0.05 — and still cannot hold a
   5% certificate on this split.
2. **Accuracy is not the certificate.** Jev at 77.7% automates two-thirds of
   traffic at a 5% joint-error budget. Laya at 37.4% with a *better* calibrated
   ECE (0.052 vs Jev's 0.092) automates none. The ranking question is coverage
   under a budget, not ECE.
3. **A trained head still wins this task, as it should.** char_lr / svm_sig
   saw BANKING77 train labels. Jev did not. The interesting comparison is Jev
   vs Laya: both zero-shot, 77-way, same ids. Jev is the first hosted decision
   model that produces a usable CRC curve on this pack.

Split-to-split (20 seeds, same 500 ids, `examples/banking77/compare-jev500/crc-split-variation.json`).
Certified coverage among feasible seeds:

| Model | @5% feasible | @5% cov mean [p10, p90] | @10% feasible | @10% cov mean [p10, p90] |
|---|---|---|---|---|
| svm_sig | 20/20 | 91.1% [83.3, 97.5] | 20/20 | 97.7% [94.8, 99.7] |
| char_lr | 20/20 | 90.4% [86.0, 96.7] | 20/20 | 98.2% [95.7, 100] |
| word_lr | 20/20 | 87.7% [78.9, 93.0] | 20/20 | 97.1% [92.8, 99.7] |
| **jev** | **20/20** | **65.4% [59.4, 70.5]** | **20/20** | **78.1% [74.9, 81.3]** |
| laya_rawT | 20/20 | 10.4% [5.4, 15.8] | 20/20 | 23.1% [14.3, 28.7] |
| laya | 10/20 | 8.6% [0.3, 16.4] | 12/20 | 13.7% [4.7, 24.2] |

Jev's 5% certificate is feasible on every split and stays in an 11-point band.
Laya's 5% certificate exists only half the time, and when it does it covers
~9%. Seed 42 (the printed compare) is typical for Jev, not a lucky draw.

This is still **this 500-row subset of this pool**, not a population. It is
enough to put Jev on the same chart as Laya and the sklearn heads, which the
toy pack was not.

**Reproduce** (live Jev calls; billed):

```bash
uv run python scripts/export_jev_predictions.py --pack banking77 --limit 500 --seed 42
# then subset the other providers onto those ids and:
uv run decisionops compare examples/banking77/providers/jev_dataset.jsonl \
  -p jev=examples/banking77/providers/jev.jsonl \
  -p laya=examples/banking77/providers/laya.jsonl \
  -p laya_rawT=examples/banking77/providers/laya_rawT.jsonl \
  -p char_lr=examples/banking77/providers/tfidf_char_lr.jsonl \
  -p word_lr=examples/banking77/providers/tfidf_word_lr.jsonl \
  -p svm_sig=examples/banking77/providers/tfidf_svm_sigmoid.jsonl \
  --target-risks 0.05,0.10 \
  --output examples/banking77/compare-jev500
```

Tracked artefacts: `examples/banking77/providers/jev_stats.json`,
`examples/banking77/compare-jev500/{compare_report.json,crc-split-variation.json}`.
The 500-row JSONL is gitignored like the rest of the pack.

---

## 9. Keep temperature on in `certify`: it is a safety net, not a coverage gain

**Decision (2026-09-27): `certify` keeps `--calibrate` as the default.** Do not
flip it to off to "save" the 20% of the calibration pool it spends fitting T.

CRC only uses the *ranking* of confidences, not whether 0.9 means 90%. So the
question is not whether calibration improves ECE (it does, §6) but whether it
changes certified coverage. Paired A/B through the product path:
`run_certification(calibrate=True)` vs `calibrate=False`, same seed so same
test split. With calibration off, the whole non-test pool goes to CRC.
Certified test coverage, 0 when infeasible; `diff` = off − on; ± is 2 SE.

Multiclass, 20 reshuffles of each fixed pool:

| Model | α | on | off | diff |
|---|---|--:|--:|--:|
| Laya, bundled T=0.10 | 5% | 3.1% (12/20 feasible) | **0.0% (0/20)** | −3.1 ± 2.1 |
| Laya, bundled T=0.10 | 10% | 4.8% (12/20) | **0.0% (0/20)** | −4.8 ± 3.2 |
| Laya, no bundled T | 5% | 12.7% | 14.1% | +1.4 ± 0.7 |
| Laya, no bundled T | 10% | 24.7% | 26.5% | +1.8 ± 0.7 |
| Jev (500) | 5% | 65.4% | 68.4% | +3.0 ± 2.3 |
| Jev (500) | 10% | 78.1% | 81.6% | +3.5 ± 1.3 |
| BANKING77 char_lr | 5% | 88.6% | 86.1% | −2.5 ± 0.7 |
| BANKING77 cnb | 10% | 59.7% | 57.1% | −2.7 ± 0.7 |
| CLINC char_lr | 5% | 81.9% | 78.3% | −3.6 ± 0.6 |

The other seven multiclass rows (svm_sig, CLINC cnb, and the remaining α) are
within ±0.8 points. Binary routing (fresh synthetic pools, 30 per row, n=500 and
2000, calibrated and overconfident): all eight rows within noise, largest
|diff| 1.2 points with a ±1.5 band.

Why keep it on:

- **The downside is lopsided.** Leaving it on costs at most ~3.5 points. Turning
  it off makes a certificate *impossible* when raw scores saturate near 1.0:
  Laya's bundled T=0.10 goes from feasible on 12/20 splits to 0/20. A customer
  whose own head does this would wrongly conclude it cannot be automated.
- **The gain from turning it off is small and not consistent.** It helps Jev and
  un-bundled Laya, hurts the sklearn heads. No simple rule (binary off,
  multiclass on) holds.
- **Binary is a wash, not a loss.** Temperature cannot reorder a binary max-prob,
  and spending 20% of the pool on T did not show up at these sizes.

An earlier session claimed the opposite ("pure loss on binary, turn it off").
That came from comparing on/off on the *same* 80% slice, which isolates the
effect of T but never tests the real default change, and from a binary
argument that was derived, not measured. Measure with this script before
revisiting.

Limits: binary pools are synthetic; multiclass is two public datasets and eight
models, reshuffled from fixed pools; only the default `temp_ratio=0.2` was
tested.

```bash
uv run python scripts/ab_certify_calibration.py            # ~1 min, offline
```

---

## 10. The label loop restores α after drift — on audit labels only

**Measured 2026-10-24.** The question: once traffic drifts after go-live, does
recertifying on the random audit sample bring the true risk back to α, and does
the obvious shortcut (recertify on the labelled escalations) fail?

Setup (`scripts/measure_label_loop.py`, 200 reps, α=5%). Certify on 3000
pre-drift cases, then drift; route 3000 drifted cases with the old profile at
audit rate ε; recertify on the audit labels only. Confidence `c ~ U(lo, 0.999)`
with `P(correct | c) = c − δ`, so **true risk is computed in closed form** (checked
against 2M Monte Carlo draws) — no test-split noise in the comparison.

- **concept drift:** same scores, accuracy down 10 points at every c (δ=0.10)
- **score drift:** still calibrated, traffic mass moves up (`lo` 0.55 → 0.75)

Mean true joint error (share above α in brackets); recertified coverage last:

| | concept | score |
|---|---|---|
| pre-drift, old λ | 0.049 (43%) | — |
| **after drift, old λ** | **0.095 (100%)** | **0.088 (100%)** |
| naive: recertify on organic escalations | 0.122 (100%) | 0.096 (100%) |
| recertify CRC, ε=5% (n≈150) | 0.046 (35%), cov 28% | 0.046 (36%), cov 59% |
| recertify CRC, ε=10% (n≈300) | 0.048 (44%), cov 29% | 0.047 (36%), cov 60% |
| recertify CRC, ε=20% (n≈600) | 0.049 (46%), cov 29% | 0.048 (43%), cov 62% |
| recertify HP 90%, ε=5% | 0.028 (6%), cov 19% | 0.027 (7%), cov 45% |
| recertify HP 90%, ε=20% | 0.039 (10%), cov 25% | 0.038 (7%), cov 55% |

What it shows:

1. **Drift breaks the live certificate.** Both drifts roughly double true risk,
   and every one of the 200 reps lands above α.
2. **Audit-only recertify restores α, from ~150 labels.** CRC mean risk returns
   to 0.046–0.049, just under α, as an in-expectation guarantee should. The
   high-probability mode keeps the share above α at 6–10%, within its 10%
   budget, at the cost of coverage.
3. **The shortcut is worse than doing nothing.** Organic escalations are the
   cases the old λ rejected, all below it. CRC on that set has never seen the
   high-confidence errors, so whenever it certified anything (200/200 reps
   under concept drift, 197/200 under score drift) the threshold was *looser*
   than the old one: 0.122 under concept drift, against 0.095 for no change.
   ~1600 labels do not help when they are the wrong ones. This is the
   measured reason `recertify` reads `Decision.audit` and nothing else.
4. **The label-free monitor cannot see concept drift.** The score distribution
   is unchanged, so `drift` said `ok` on 190/200 reps while true risk doubled.
   Score drift was flagged on 200/200. Drift monitoring is not a substitute for
   the audit sample; only labels reveal a model that got worse at the same
   confidence.

A regression test pins point 2 with teeth:
`tests/test_recertify.py::test_label_loop_restores_alpha_after_concept_drift`
uses ~100 audit labels, where the CRC penalty moves mean risk by ~0.010
(measured 0.047 vs 0.057 without it), so deleting the penalty fails the test.

Limits: synthetic binary pools with one auto choice, two drift shapes, α=5%
only. The audit sample here is the decision log's own `audit=True` rows; real
labels arrive late and noisy, which this does not model.

```bash
uv run python scripts/measure_label_loop.py            # 200 reps, ~40 s, offline
```

---

## Limits on all of the above

**Laya scope.** One checkpoint, zero-shot, one task shape, 77/151 options. Its
authors describe it as a fine-tuning base, so weak zero-shot accuracy is expected
behaviour, not a defect. No fine-tuning was performed. Nothing here predicts how
a fine-tuned Laya behaves.

**Jev scope.** One hosted snapshot (`jev-1.13.0`), zero-shot, 500 of 3076
BANKING77 test rows, no contract descriptions. Nothing here predicts a
fine-tuned Jev, a binary Noul routing task, or production traffic. 1% CRC was
not run: `n_crc_cal=120` is too close to the 99-sample floor.

**sklearn heads are stress-test baselines.** TF-IDF + linear models, chosen
because their probability behaviour differs sharply. Not recommendations.

**Public benchmarks, not production traffic.** Real workloads bring label noise,
drift, and delayed outcomes that these datasets do not.

**§3 is a single-pool result.** `scripts/resample_crc.py` reshuffles the same
3076-row BANKING77 held-out set 40 times. That measures split-to-split variation
*inside this pool*, not a distributional expectation — the same trap that
produced a retracted "validated across 40 seeds" claim earlier. The qualitative
ordering (sklearn heads automate at 1%; Laya essentially does not) is stable
across those splits. The point estimates are not portable beyond this pool.

**Verdicts rest on a certificate plus one policy number.** Since 2026-09-27,
`gate` decides `go` / `watch` / `no-go` from a CRC certificate at
`--verdict-alpha` (5%): no certificate → no-go; certified test coverage below
`--min-coverage` (30%) → watch; otherwise go. Feasibility is certified; the 30%
bar is a hand-picked policy choice, not theory. The old rule needed an ECE drop
for `go`, which on BANKING77 gave Laya (certifies nothing) and Jev (67% at 5%)
the same `watch`. The committed reports under `examples/*/gate*/` were
regenerated under the new rule on the same inputs.

**One split, one draw.** Each committed gate report is a single seed-42 split.
BANKING77 char_lr's certified test joint error came out 0.0627, above its 5%
target. That is expected under CRC (about half of single splits land above α,
§5) and does not void the certificate; it is also why a per-split `go` is a
screening signal, not a guarantee on that split.
