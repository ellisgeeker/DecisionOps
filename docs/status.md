# Project status

**Last updated:** 2026-09-26
**Version:** 0.1.0a1 (unreleased)

Where the project actually is, what to pick up next, and what is known to be
incomplete. Written so it is usable after a gap of days or weeks.

---

## Pick up here tomorrow

Roadmap item 4: cost + offline cascade simulation. Contract `costs` are
already parsed and validated but not used in any report yet.

Items 1–3 are done: README/status positioning, `contract.py`, and
`--contract` on gate / compare / certify with asymmetric risk.

## Product positioning (decided 2026-09-26)

- **DecisionOps certifies the decision layer, not the execution layer.** The
  certified model is a cheap decision model (Jev, Laya, a customer's own head)
  that outputs a closed-set choice with probabilities. Claude / Qwen / DeepSeek
  sit *behind* it as expensive executors, called only on escalation. Their
  chat-style confidence is not what gets calibrated.
- **The core question is routing:** "is this case worth sending down the slower,
  more expensive path?" The certified claim is: at miss rate ≤ α, this many
  expensive calls are saved.
- **Risk is asymmetric.** Only errors on the automatic (cheap) path are risk.
  Escalating a case that did not need it costs money, not safety.
- **Code review is one showcase scenario, not the product.** Example: does this
  diff need a deep LLM review? The same contract describes ticket triage, risk
  review, and agent tool-call confirmation. Scenario specifics live in
  `examples/`, never in the core.

## Roadmap (ordered)

1. ~~Positioning in README / status~~ — done.
2. ~~Decision contract schema + validator~~ — done: `src/decisionops/contract.py`.
3. ~~Asymmetric risk~~ — done: with `--contract`, selective / CRC / gate /
   compare count only auto-path errors. The loss
   `1{auto-eligible ∧ conf ≥ λ ∧ wrong}` stays monotone and bounded, so the CRC
   guarantee is unchanged. Closed-form test in `tests/test_contract.py`: the
   asymmetric rule lands +0.010 above λ*; a symmetric rule leaking escalation
   errors lands +0.258 above and fails. No contract = previous behaviour.
4. **Cost + offline cascade simulation** (pulled forward from V0.5) — the
   "cost saved vs miss rate" curve, which is the headline value.
5. **Certified profile + `decide()`** — turn an offline certificate into a
   runtime call: `decide(prediction, profile) → AUTO / ESCALATE`.
6. **Code review example pack** — small synthetic data, labelling guide (where
   ground truth comes from: human-confirmed deep-review findings, reverts,
   incidents), one end-to-end report.
7. **Decision-backend adapters** — generic HTTP provider (any model returning
   choice + probabilities), then TypeSafe Jev.
8. **Execution-layer LLMs** (Claude / Qwen / DeepSeek) — escalation targets in
   cascade / gateway only, never the calibration target.
9. **Thin gate** — GitHub Action first, then a gateway (`POST /v1/decision`).
10. **Wrap-up** — v0.1.0 release, CRC `confidence_level` API, outcome feedback,
    drift, recertification.

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

- **70 tests passing**, ruff clean.

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

See **Roadmap** above for the ordered work. Beyond it:

- **Fine-tune Laya and re-measure** — model research, not on the product path.
- **Isotonic in `compare` / `certify`** — gate has the contrast; CRC still uses
  temperature only, intentional for V1.
- **"LLM as decision model" adapter** — possible later as a compatibility path,
  but reports must mark it uncalibrated.

M0 (git + Apache-2.0 LICENSE + GitHub Actions + 85% coverage gate) is done.
Large regenerated HF JSONL packs are gitignored; rebuild with
`scripts/prepare_hf_intent_gates.py`. Remote GitHub repo not created yet.
