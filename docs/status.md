# Project status

**Last updated:** 2026-09-26
**Version:** 0.1.0a1 (unreleased)

Where the project actually is, what to pick up next, and what is known to be
incomplete. Written so it is usable after a gap of days or weeks.

---

## Pick up here tomorrow

**OpenAI-compatible adapter** (DeepSeek etc.). Needs a probability-source
decision: `logprobs`/`top_logprobs`, structured self-reported probabilities, or
repeated sampling. The first is most credible; none has been tested. Requires
an API key and a cost budget.

Alternatively: fine-tune Laya and re-measure on the same banking77/clinc
protocol — the largest open question about the model.

---

## Done

### Core pipeline

| Component | Location | Notes |
|---|---|---|
| Schemas | `schemas.py` | `DecisionCase`, `DecisionPrediction`, `EvalResult`, `GateReport`, `ComparisonReport` |
| JSONL IO | `io.py` | Loads datasets/predictions, joins on `id`, rejects gaps and duplicates |
| Metrics | `metrics/` | Accuracy, ECE (fixed bins), multiclass Brier, NLL, latency percentiles |
| Cal/test split | `split.py` | Seeded, reproducible; guarantees ≥2 rows per side |
| Temperature scaling | `calibration/temperature.py` | Scalar `T` by NLL minimisation over `exp` parameterisation |
| Isotonic regression | `calibration/isotonic.py` | Binary map confidence→correctness; third gate condition; argmax preserved |
| Selective prediction | `selective/` | Threshold sweep, risk-coverage, held-out threshold application |
| Conformal Risk Control | `conformal/` | Angelopoulos et al.; validated (see below) |
| Model comparison | `compare.py` | Shared split; ranks by CRC-certified coverage; empirical `!` kept as contrast |
| Reports | `report/` | Rich console + JSON + self-contained HTML (reliability + risk-coverage SVG) |

### CLI commands

| Command | Status |
|---|---|
| `decisionops eval` | done — same-set smoke; `--provider static\|laya` |
| `decisionops gate` | done — cal/test split; temperature + isotonic when `--calibrate`; `--provider static\|laya` |
| `decisionops compare` | done — shared split, CRC ranking, empirical table retained with `!` |
| `decisionops certify` | done — CRC thresholds; `--provider static\|laya` |

### Providers

| Provider | Location | Status |
|---|---|---|
| Static file | `adapters/base.py` | done — offline predictions, used for CI |
| Laya | `adapters/laya.py` | done — local inference; `--provider laya` on eval/gate/certify |
| sklearn heads | `scripts/export_multi_providers.py` | done — 4 heads with deliberately different probability behaviour |
| OpenAI-compatible | — | not started |

The Laya adapter imports upstream `rl_common.py` from the downloaded snapshot
rather than vendoring a copy, so the sequence layout and temperature buckets stay
in sync with the published checkpoint.

### Datasets

| Pack | n | Source |
|---|--:|---|
| `examples/toy-routing/` | 120 | hand-built smoke test |
| `examples/gate-synthetic/` | 2000 | offline synthetic, sklearn probs |
| `examples/banking77/` | 3076 held-out | `mteb/banking77` |
| `examples/clinc_oos/` | 8600 held-out | `clinc/clinc_oos` config `plus` |

Both public packs fit their classifier on the HF **train** split and export only
**held-out** rows, so no train leakage enters the DecisionOps pool. Their
`*.jsonl` files are gitignored (rebuild with
`scripts/prepare_hf_intent_gates.py`); summaries and gate/compare reports stay
in tree.

### Validation performed

- **55 tests passing**, ruff clean.

- CRC correctness is now a **regression test** (`tests/test_crc_analytic.py`):
  60 fresh pools, n_cal=150 so `B/(n+1)` is a visible share of α, `λ̂` compared
  to a closed-form `λ*`. A third test deletes the penalty and asserts that the
  joint check then *fails* — otherwise the first two have no teeth. The 300-rep
  script `scripts/diagnose_crc.py` remains for the two acceptance conventions.
- Split-to-split CRC variation on the BANKING77 pool is recorded in
  `report/crc-split-variation.json` (40 seeds, 5 models). Laya is infeasible at
  1% on 34/40 seeds; sklearn heads are feasible on all 40. See
  [findings.md](findings.md) §3. This estimates **the pool**, not a population.
- Isotonic gate contrast (toy smoke): ECE raw 0.085 → temperature 0.082 →
  isotonic 0.047; accuracy unchanged. Verdict still focuses on temperature.

---

## Not done

### Next up (small, unblocked)

1. **OpenAI-compatible adapter (DeepSeek etc.).** Needs a probability-source
   decision (see "Pick up here").

M0 (git + Apache-2.0 LICENSE + GitHub Actions + 85% coverage gate) is done.
Large regenerated HF JSONL packs are gitignored; rebuild with
`scripts/prepare_hf_intent_gates.py`. Remote GitHub repo not created yet.

### Later (larger)

2. **Cascade simulation.** Offline only. Needs at least two providers with
   genuinely different risk-coverage curves.
3. **Fine-tune Laya and re-measure.** The single largest open question about the
   model (see Known gaps).
4. **Production gateway, outcome feedback, drift, recertification.** Deliberately
   deferred. Each depends on an outcome pipeline that does not exist.
5. **Isotonic in `compare` / `certify`.** Gate has the contrast; comparison entry
   and CRC still use temperature only — intentional for V1.
