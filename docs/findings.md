# Findings

**Last updated:** 2026-09-25

Measured results, with the protocol and limits of each. Numbers here came from
runs on public benchmarks; nothing is synthetic unless labelled so.

---

## Setup

| | |
|---|---|
| Datasets | BANKING77 (`mteb/banking77`, 3076 held-out rows, 77 classes), CLINC OOS `plus` (`clinc/clinc_oos`, 8600 held-out rows, 151 classes) |
| Models | Laya `convaiinnovations/laya` (zero-shot) + 4 sklearn heads |
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

## Limits on all of the above

**Laya scope.** One checkpoint, zero-shot, one task shape, 77/151 options. Its
authors describe it as a fine-tuning base, so weak zero-shot accuracy is expected
behaviour, not a defect. No fine-tuning was performed. Nothing here predicts how
a fine-tuned Laya behaves.

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

**Verdicts are heuristics.** `go` / `watch` / `no-go` use hand-picked thresholds
with no theory behind them.
