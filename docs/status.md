# Project status

**Last updated:** 2026-09-27 (`decide()` landed; next is the code review example pack)
**Version:** 0.1.0a1 (unreleased)

Where the project actually is, what to pick up next, and what is known to be
incomplete. Written so it is usable after a gap of days or weeks.

**Session handoff (2026-09-27):** Roadmap 1–6 done. Live Jev verified. Offline
cascade and the runtime `decide()` rule are in. Next chat should start at
roadmap #7 (code review example pack).

---

## Pick up here tomorrow

**Next: roadmap #7 — code review example pack.** Small synthetic data, a
labelling guide (ground truth from human-confirmed deep-review findings,
reverts, incidents), and one end-to-end run: `certify --profile` → `cascade`
→ `decide`, using `examples/contracts/routing.json` as the starting contract.

Just finished — roadmap #6, certified profile + `decide()` (`src/decisionops/decide.py`):

- `CertificationProfile` freezes one α of a `CertificationReport`: λ̂, T,
  provider/model, choices + auto set, `contract_hash`, and the evidence
  (n_cal, CRC bound, cal coverage, test stats, split).
  `certify --profile p.json [--profile-alpha A]` writes it; an infeasible α
  writes nothing and exits 1.
- `decide(prediction, profile)`: apply T to the **raw** probabilities, then AUTO
  iff the choice is automatic and conf ≥ λ̂, else ESCALATE. That is the CRC
  acceptance rule, so the bound carries over. The `reason` field is one of
  `certified` / `below_threshold` / `escalate_choice` / `outside_contract`.
- Mismatch guards: a prediction tagged with another `model` raises
  `ProfileMismatchError`, and so does a contract whose `fingerprint()` differs.
  The fingerprint covers choices, auto set, descriptions, instructions and
  yes_choice; changing costs does not invalidate it. The file has a
  `schema_version`.
- `decisionops decide PROFILE -p preds.jsonl [--contract] [-o decisions.jsonl]`
  is the offline smoke. Toy pack @10%: T=1.137, λ̂=0.853, 52/120 AUTO.
- Regression test `test_decide_reproduces_certified_test_numbers`: `decide()`
  on the held-out split must reproduce the certificate's `n_accepted` and
  joint error exactly.
- Not covered: isotonic (CRC uses temperature only), the Laya builtin
  temperature flag (not recorded in the profile), and drift or recertification
  triggers (roadmap 11).

Context from earlier work:

1. ~~Live Jev smoke + BANKING77 500~~ — done 2026-09-27. Toy Choice still 1.0 /
   ~380 ms; Noul still routes skip vs deep-review. On 500 BANKING77 ids (same
   rows as Laya + sklearn): Jev acc 0.794, ECE already ~0.09, CRC @5%
   coverage 65% [59, 71] across 20 splits. Laya on those ids still cannot
   hold 5%. Numbers in [findings.md](findings.md) §8.
2. ~~Cost + offline cascade simulation~~ — done 2026-09-27.
   `decisionops cascade` + `src/decisionops/cascade.py`. Cost-saved vs joint
   miss-rate on the held-out split; escalate priced with an oracle (true
   escalate label, else cheapest escalate choice). Thresholds are empirical —
   pair with `certify` for a CRC bound. Example contract:
   `examples/contracts/routing.json`. Tests: `tests/test_cascade.py`.

Decided 2026-09-26: customers bring their own decision model (Jev, a fine-tuned
Laya, or their own head). DecisionOps does not train one, so `train-screener`
is dropped from the roadmap.

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
4. ~~`train-screener`~~ — **dropped** (2026-09-26). Customers bring their own
   decision model; a customer without one is not a customer. The sklearn heads
   in `scripts/` stay as internal demo / benchmark fixtures only.
5. ~~**Cost + offline cascade simulation**~~ — done 2026-09-27
   (`decisionops cascade`, `cascade.py`). Cost-saved vs joint miss-rate curve
   on the held-out split; oracle escalate pricing; contract must supply costs.
6. ~~**Certified profile + `decide()`**~~ — done 2026-09-27 (`decide.py`,
   `certify --profile`, `decisionops decide`).
7. **Code review example pack** — small synthetic data, labelling guide (where
   ground truth comes from: human-confirmed deep-review findings, reverts,
   incidents), one end-to-end report.
8. **Decision-backend adapters** — ~~TypeSafe Jev~~ done (`--provider jev`,
   Choice + Noul; live-verified 2026-09-27, including a 500-case BANKING77
   compare). Next: a generic HTTP provider for customers' own heads. Offline
   JSONL already works for any model.
9. **Execution-layer LLMs** (Claude / Qwen / DeepSeek) — escalation targets in
   cascade / gateway only, never the calibration target.
10. **Thin gate** — GitHub Action first, then a gateway (`POST /v1/decision`).
11. **Wrap-up** — v0.1.0 release, CRC `confidence_level` API, outcome feedback,
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
| Decision contract | `contract.py` | Closed-set choices, auto-set, costs, yes_choice; asymmetric risk for CRC/gate |
| Offline cascade | `cascade.py` | Cost-saved vs joint miss-rate curve; oracle escalate pricing; needs contract costs |
| Runtime decide | `decide.py` | `CertificationProfile` from a certify report; `decide()` → AUTO/ESCALATE; model + contract-hash guards |
| Reports | `report/` | Rich console + JSON + self-contained HTML (reliability, risk-coverage, cascade cost curve) |

### CLI commands

| Command | Status |
|---|---|
| `decisionops eval` | done — same-set smoke; `--provider static\|laya\|jev` |
| `decisionops gate` | done — cal/test split; temperature + isotonic when `--calibrate`; `--provider static\|laya\|jev` |
| `decisionops compare` | done — shared split, CRC ranking, empirical table retained with `!`; offline JSONL only |
| `decisionops certify` | done — CRC thresholds; `--provider static\|laya\|jev`; `--profile` writes a runtime profile |
| `decisionops cascade` | done — cost-saved vs miss-rate; requires `--contract` with costs |
| `decisionops decide` | done — route a predictions JSONL with a profile; offline smoke |

Provider-specific flags (`--model-dir`, `--no-builtin-temperature`, `--jev-model`,
`--jev-base-url`) are rejected when combined with a different `--provider`.

### Providers

| Provider | Location | Status |
|---|---|---|
| Static file | `adapters/base.py` | done — offline predictions, used for CI |
| Laya | `adapters/laya.py` | done — local inference; `--model-dir` for a fine-tuned checkpoint, `--no-builtin-temperature` |
| TypeSafe Jev | `adapters/jev.py` | done — hosted Choice + Noul via `typesafe-sdk==0.7.2` (`--extra jev`); `--jev-model` (incl. fine-tuned ids), `--jev-base-url`; export via `scripts/export_jev_predictions.py`. Live-verified 2026-09-27 against `jev-1.13.0` (Choice + Noul). BANKING77 n=500 compare in [findings.md](findings.md) §8 |
| sklearn heads | `scripts/export_multi_providers.py` | done — 4 heads with deliberately different probability behaviour |

The Laya adapter imports upstream `rl_common.py` from the downloaded snapshot
rather than vendoring a copy, so the sequence layout and temperature buckets stay
in sync with the published checkpoint.

Jev supports **Choice** (default) and **Noul** (`--jev-question noul`). Noul's
P(yes) maps onto a binary contract via `yes_choice`, which is the natural form
for routing ("does this need a deep review?"); descriptions become Noul's
`true` / `false` criteria. **Score** is out of scope: its value is the ordering
of its levels, so treating levels as unordered classes would make ECE, accuracy
and the CRC loss misleading. Supporting it needs an ordinal risk design (e.g.
"under-rating by more than one level is an error"), not just an adapter.

Earlier note (2026-09-26) claimed Noul "does not map onto a case's choice set".
That was wrong: a binary case maps exactly, as above.

Contract `descriptions` / `instructions` are sent to live providers as model
input (Jev `criteria`; Laya renders them as `choice: text`). A certificate holds
only for the input the model saw, so reports record them under
`metadata.contract` and any change means recertify.

OpenAI-compatible LLMs (Claude / Qwen / DeepSeek) are **not** decision providers:
they are the execution layer, called on escalation (roadmap item 9).

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

- **108 tests passing** (1 live Jev smoke skipped without a key), 91% coverage,
  ruff clean.

- CRC correctness is now a **regression test** (`tests/test_crc_analytic.py`):
  60 fresh pools, n_cal=150 so `B/(n+1)` is a visible share of α, `λ̂` compared
  to a closed-form `λ*`. A third test deletes the penalty and asserts that the
  joint check then *fails* — otherwise the first two have no teeth. The 300-rep
  script `scripts/diagnose_crc.py` remains for the two acceptance conventions.
- Split-to-split CRC variation on the BANKING77 pool is recorded in
  `report/crc-split-variation.json` (40 seeds, 5 models). Laya is infeasible at
  1% on 34/40 seeds; sklearn heads are feasible on all 40. See
  [findings.md](findings.md) §3. This estimates **the pool**, not a population.
- Jev vs Laya vs sklearn on a **shared 500-id BANKING77 subset** (20 seeds,
  `n_crc_cal=120`): Jev @5% feasible 20/20, mean coverage 65.4%; Laya
  feasible 10/20, mean 8.6% when it exists. See [findings.md](findings.md) §8
  and `examples/banking77/compare-jev500/`.
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
