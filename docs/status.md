# Project status

**Last updated:** 2026-10-24 (label loop measured, findings §10; next is release 0.1.1)
**Version:** 0.1.0 (tagged and pushed 2026-10-24; not on PyPI; recertify/drift/shadow unreleased on main)

Where the project actually is, what to pick up next, and what is known to be
incomplete. Written so it is usable after a gap of days or weeks.

**Session handoff (2026-09-27):** The V1 *library* path is done (roadmap 1–6
and 8: contract, cascade, `decide()`, Jev, HTTP). The thin gate (GitHub
Action + `POST /v1/decision`), execution-layer LLMs, and the code review
example pack are one *production/showcase* bundle — they share the PR-routing
story and wait until after v0.1. Next product work is roadmap #11 (release).

---

## Pick up here tomorrow

**Next: release 0.1.1** (recertify / drift / shadow guide / loop measurement
are on `main`, unreleased). Then certified conditional (selective) risk or the
code-review showcase pack.

Done 2026-10-24 — label loop measured end to end ([findings.md](findings.md) §10):

- `scripts/measure_label_loop.py`, 200 reps, α=5%, closed-form true risk.
  Drift doubles true risk of the old λ (0.049 → 0.095 concept, 0.088 score;
  100% of reps above α). Audit-only CRC recertify restores mean risk to
  0.046–0.049 from ~150 labels; HP 90% keeps the share above α at 6–10%.
- Recertifying on organic escalations is worse than no change (0.122 under
  concept drift): those rows sit below the old λ, so CRC certifies a looser
  threshold. Measured justification for audit-only.
- `drift` misses concept drift (ok on 190/200) and catches score drift
  (200/200). Label-free monitoring does not replace the audit sample.
- Regression test with teeth:
  `test_label_loop_restores_alpha_after_concept_drift` fails when the CRC
  penalty is deleted (mean 0.0576 vs bar 0.0515).

Done 2026-10-24 — published: public repo `github.com/ellisgeeker/DecisionOps`
(SSH alias `github-ellisgeeker`), `main` + tag `v0.1.0`. History rewritten
before the first push: author → `ellisgeeker`, `decisionops-plan.md` dropped
(now gitignored, local only), local absolute paths made repo-relative (export
scripts use `_rel()`). PyPI undecided. CI result not checked from here (`gh`
not logged in); Python 3.12 suite passed locally.

Done 2026-10-24 — shadow-mode guide (docs, 0.1.x):

- [`docs/shadow-mode.md`](shadow-mode.md): why shadow/backfill labels, sample
  floors, cutover checklist, CLI copy-paste for certify → decide → drift →
  recertify.
- `scripts/run_shadow_mode_demo.py`: offline BANKING77 walkthrough (default
  writes `examples/banking77/shadow-demo/`). Smoke @800 rows, α=5%, ε=5%:
  certificate feasible, 36 audit labels, drift `ok` on same-run split,
  recertify feasible.
- BANKING77 is a practice pack (all intents automatic in the demo contract),
  not a production risk claim.

Done 2026-10-24 — label-free drift monitor (0.1.x):

- `run_drift_monitor(live, baseline, profile=…)` compares two `decide()`
  windows without labels. Metrics: certified rate (`reason in {certified,
  audit}`, audit-invariant and comparable to coverage_calibration), auto
  rate, confidence PSI (equal-width bins) and two-sample KS.
- Verdicts: `ok` / `watch` / `recertify_suggested` with fixed policy bands
  (rate 5%/10%, PSI 0.10/0.25, KS p 0.05/0.01). A shift is an alarm, not a
  certificate.
- CLI: `decisionops drift LIVE --baseline BASE [--profile P]`; exit 0/1/2.
- Tests: PSI sanity, matched windows stay ok, confidence shift alerts, CLI
  exit codes.

Done 2026-10-24 — outcome feedback + `recertify` (0.1.x):

- `OutcomeFeedback` schema (`id`, `label`, `label_source`, `matured_at`) and
  JSONL loaders `load_outcomes` / `load_decisions`.
- `run_recertification` joins decisions × outcomes × predictions, keeps only
  `Decision.audit=True` rows (organic escalations ignored), optional
  `--as-of` / `--since` maturity window, applies the **frozen** profile
  temperature (never re-fits T on audit mass), runs CRC at the profile alpha.
- Verdict `replace` / `keep` / `infeasible` with threshold/coverage delta vs
  the live profile; `--profile-out` writes a child profile tagged
  `split.protocol=audit_recertify`.
- CLI: `decisionops recertify PROFILE --decisions … --outcomes … -p …`.
- Tests: organic labels excluded, maturity filter, matching traffic stays
  feasible, drifted audit tightens or fails, CLI round trip.

Done 2026-10-24 — v0.1.0 (local):

- Top-level public API in `decisionops/__init__.py` (`decide`,
  `run_certification`, `load_profile`, `DecisionContract`, …); pinned by
  `tests/test_public_api.py`. Deeper module paths are not promised stable.
- Deleted dead `report/json_report.py` (the live `write_json_report` is in
  `report/__init__.py`). Version `0.1.0`, `CHANGELOG.md`, README Python API
  section.
- Wheel + sdist build; the wheel installs in a clean venv and runs
  `certify --profile` → `decide` on the toy pack.

Re-ordered plan (2026-10-23), algorithm and loop work before real traffic:

1. ~~Laya builtin-temperature guard~~ — done 2026-10-23.
2. ~~Weighted-loss CRC~~ — done 2026-10-23, below.
3. ~~v0.1.0 release~~ — done locally 2026-10-24.
4. ~~Feedback format + `recertify` from audit rows only~~ — done 2026-10-24
   (drift simulation in `tests/test_recertify.py`).
4b. ~~Label-free drift monitor~~ — done 2026-10-24 (`decisionops drift`).
5. Certified conditional (selective) risk via Learn-Then-Test; certified
   cascade thresholds (today the cascade curve is empirical, two-stage,
   oracle-priced).
6. Score (ordinal) risk design; the #7/#9/#10 showcase bundle.

Real-traffic validation follows the loop, so it tests the whole mechanism.
Isotonic in CRC is **not** planned: it is monotone, so it does not reorder
confidence, and CRC depends only on the order.

Done 2026-10-23 — weighted (severity) loss in CRC:

- Contract field `severity: {true_label: weight}` is separate from `costs`.
  A miss on the automatic path contributes `severity.get(y, 1.0)`; omitted
  labels default to 1.0; empty map is plain 0-1 loss. `B = max severity`
  (or 1). Keys must be contract choices; values must be > 0.
- Granularity is **per true label**, not per prediction: the motivating case
  is a single auto choice (`skip`) where "needed a human" and "needed an LLM"
  differ only in the true label. Decided after first proposing per-prediction
  severity and noticing it collapses on that contract.
- `fingerprint()` covers severity (changing it invalidates a profile); costs
  still do not. `run_certification` derives `loss_bound` from the contract
  unless an explicit override is passed. High-probability mode still requires
  0-1 loss and rejects fractional / non-unit severity.
- Controlled quantity becomes `weighted_joint_automated_error` when severity
  is set. Example: `examples/contracts/code-review-routing.json`
  (`human_review: 5`, `deep_llm_review: 1`).
- Tests: schema + fingerprint, CRC raises the bar when a high-severity miss
  sits at the same confidence as a low-severity one, HP refuses weighted
  loss, `run_certification` picks up B from the contract.

Done 2026-10-23 — Laya builtin-temperature guard:

- Bug: the profile did not record whether Laya's bundled temperature was on.
  The setting lived only in per-prediction `metadata`, and `model` was the
  same either way, so a profile certified with it on silently accepted
  predictions made with it off — a threshold applied to different
  probabilities.
- Fix: `laya_model_id()` appends `+no-builtin-temperature` to the model id
  when it is off, so `decide()`'s existing model guard refuses the cross-over.
  Default (on) ids are unchanged, so existing on-mode profiles still load.
- Tests: `tests/test_laya_model_id.py` (runs in CI, no torch) and an extra
  assertion in the torch-gated `tests/test_laya_adapter.py`.
- Committed `gate-laya_rawT` reports were produced with it off and still say
  `rl-agent`; they are reports, not profiles, so nothing breaks. Regenerate
  only if they are rebuilt for another reason.

Done 2026-10-23 — roadmap #11.1, audit sampling in `decide()`:

- `CertificationProfile.audit_rate` (ε, default 0 = off) and `audit_seed`;
  `certify --profile … --audit-rate 0.05 [--audit-seed N]` writes them. Both
  are part of `profile_hash`.
- Membership is `sha256(f"{seed}:{id}") < ε`: independent of the score,
  reproducible, no state across processes. Every drawn case gets
  `Decision.audit=True`, *including* ones that escalate anyway, so the
  `audit=True` rows are a uniform sample of traffic — recertify on exactly
  those. A drawn case the rule would AUTO becomes ESCALATE,
  `reason="audit"`. Forced escalations only remove automated errors, so the
  bound still holds.
- Tests (`tests/test_decide.py::test_audit_*`): rate and reproducibility,
  KS test that audit confidence matches all traffic while organic
  escalations do not, audit only removes auto errors, CLI round trip.
- Toy pack @10%, ε=5%, seed 1: 50/120 AUTO (was 52), 3 in the audit sample.
- Caveat: the draw keys on `id`, so ids must be unique per case. A caller
  that reuses ids (retries, replays) gets the same audit decision each time,
  which is intended for retries but means duplicate ids are not resampled.
  `decisionops decide` warns when the input has repeated ids.

Done 2026-10-23 — pre-commit review of the 09-27 work, two fixes:

- `DecisionPrediction` rejects NaN / inf probabilities. `json.loads` accepts
  `NaN`, which passed the non-negative check and made confidence NaN, so a
  bad HTTP / JSONL row silently escalated instead of failing.
- HTTP provider refuses redirects. urllib turns a 302 on POST into a GET and
  copies all headers, so the Bearer token would follow `Location` to any host.

Done 2026-09-27 — `gate` verdict now comes from a certificate:

- `run_gate` runs `run_certification` at `--verdict-alpha` (5%) with the same
  seed / cal_ratio / temp_ratio / contract as `certify`, so both agree. no-go
  = infeasible, watch = certified coverage < `--min-coverage` (30%), go
  otherwise. ECE is context only. Tiny pools that cannot split → no-go.
- BANKING77 now: svm_sig / char_lr / cnb / Jev → go (90.5 / 91.5 / 45.1 /
  67.1% certified), Laya → no-go. Old rule gave Laya and Jev both `watch`.
- `GateVerdict` dropped `coverage_in_band_1pct` (the 15–75% band) and gained
  `certified_*`, `verdict_alpha`, `min_coverage`.
- Committed gate reports under `examples/` regenerated with the same inputs
  and seed 42 (temperatures match the old reports to 1e-9), now with HTML:
  BANKING77 char_lr go (91.5%), Laya no-go (was watch), Laya raw-T 800 watch
  (10.2%), CLINC word_lr go (76.7%, was watch). `gate-laya_rawT` is the seed-42
  800-row subset from `export_laya_predictions._subsample`; the JSONL on disk is
  now the full 3076-row export.
- `certify` keeps temperature on by default; measured in
  [findings.md](findings.md) §9.

Done 2026-09-27 — `confidence_level` (high-probability certificates):

- `certify_threshold(..., confidence_level=0.95)` / `certify
  --confidence-level 0.95` issue an RCPS certificate with a Clopper-Pearson
  bound: P(true joint error ≤ α) ≥ 0.95 over the calibration draw. Default
  (`None`) is unchanged CRC, which bounds only E[loss].
- Fixed-sequence scan from the strictest λ down, so no multiplicity
  correction. 0-1 loss only (`loss_bound` must be 1). Zero-error sample
  floor is `ln(δ)/ln(1−α)`: 299 at α=1%, 95% (CRC: 99).
- `CRCResult`, report metadata and `CertificationProfile` carry
  `guarantee` + `confidence_level`; console / HTML / `decide` output say
  which guarantee was issued.
- Measured on 300 fresh calibrated pools (n_cal=300, α=5%), rate of
  certificates whose true risk exceeds α: HP 0.140 at δ=0.20, 0.030 at
  δ=0.05; plain CRC 0.417. Regression test:
  `tests/test_crc_high_probability.py::test_hp_violation_rate_is_at_most_delta`.
- `scipy` added to `dependencies` (it was imported directly but only present
  via scikit-learn). CI now installs `--extra jev`: without it the Jev adapter
  tests skip and coverage was 84.2%, under the 85% gate.

Do **not** start the thin gate or the code review pack next. The original
plan said the first version is Eval, not a Gateway. The GitHub Action is
the code-review integration (run `decide` on a PR); `POST /v1/decision`
is the same runtime in HTTP form. Both belong with the showcase (#7) and
execution-layer LLMs (#9) *after* the library is released, so the demo
can run `certify → cascade → decide → gate` for real. Starting contract
when that bundle starts: `examples/contracts/routing.json`.

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
  The fingerprint covers choices, auto set, severity, descriptions,
  instructions and yes_choice; changing costs does not invalidate it. The
  file has a `schema_version`.
- `decisionops decide PROFILE -p preds.jsonl [--contract] [-o decisions.jsonl]`
  is the offline smoke. Toy pack @10%: T=1.137, λ̂=0.853, 52/120 AUTO.
- Regression test `test_decide_reproduces_certified_test_numbers`: `decide()`
  on the held-out split must reproduce the certificate's `n_accepted` and
  joint error exactly.
- Not covered: isotonic (CRC uses temperature only; not planned, see above)
  and drift or recertification triggers (roadmap 11). The Laya builtin
  temperature flag is now in the model id (2026-10-23).

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
  more expensive path?" The certified claim is: at an *expected* miss rate
  ≤ α, this many expensive calls are saved.
- **Default guarantee is CRC, in expectation (decided 2026-09-27).** Measured
  on 300 fresh pools (α=5%): mean true risk 4.83%, but 41.7% of individual
  certificates sit above α (p95 7.0%). Never phrase it as "each certificate
  ≤ α". `--confidence-level 0.95` is the opt-in per-certificate guarantee
  (3.0% above α), costing coverage 45.9% → 35.2% and ~3× the calibration
  data at α=1%. Revisit the default only if a customer needs it for audit.
- **Risk is asymmetric.** Only errors on the automatic (cheap) path are risk.
  Escalating a case that did not need it costs money, not safety.
- **Certificates are recertified on unbiased live samples, on a schedule.**
  `decide` draws the audit sample; `recertify` rebuilds CRC on matured audit
  labels and is the primary defence. `drift` watches behaviour without labels
  and can only bring a recertify forward: it cannot see concept drift (ok on
  190/200 while risk doubled, findings §10). Organic escalations stay out of
  the calibration set. Open problems: *Label strategy*.
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
7. **Code review example pack** — *showcase, with #9/#10, after v0.1.*
   Small synthetic data, labelling guide (ground truth from human-confirmed
   deep-review findings, reverts, incidents), one end-to-end report. Not a
   stand-in for missing library work.
8. **Decision-backend adapters** — ~~TypeSafe Jev~~ done (`--provider jev`,
   Choice + Noul; live-verified 2026-09-27, including a 500-case BANKING77
   compare). ~~Generic HTTP provider~~ done (`--provider http`, stdlib
   `urllib`, gold `label` never sent). Offline JSONL already works for any
   model.
9. **Execution-layer LLMs** (Claude / Qwen / DeepSeek) — escalation targets
   in the showcase gate only, never the calibration target. Same bundle as
   #7/#10.
10. **Thin gate** — *same bundle as #7, after v0.1.* GitHub Action (PR
    routing — this *is* the code-review integration) then
    `POST /v1/decision`. V1 is the CLI; the original plan deferred Gateway.
11. **Wrap-up** — in progress. ~~CRC `confidence_level` API~~ done
    2026-09-27. Remaining, in order:
    1. ~~**Audit sampling in `decide()`**~~ — done 2026-10-23.
    2. ~~**Weighted-loss CRC**~~ — done 2026-10-23 (`contract.severity`).
    3. ~~**v0.1.0 release**~~ — done locally 2026-10-24; remote repo pending.
    4. **0.1.x: closing the label loop** — ~~outcome feedback + `recertify`~~,
       ~~drift monitor~~, ~~shadow-mode guide~~ done 2026-10-24. Open problems
       (window vs staleness, low volume, IPW) remain. See *Label strategy*.
    The #7/#9/#10 bundle is the public demo after the release.

## Label strategy (decided 2026-09-27)

A certificate needs labels that are exchangeable with live traffic. That is
the product's main unsolved risk: once a cheap path is live, auto-accepted
cases never get the expensive verdict, so organic labels come only from
escalated (low-confidence) cases — a biased sample. Plan:

- **First certificate: backfill or shadow.** A prospective customer is, by
  definition, sending everything down the expensive path today, so their
  history already labels the full distribution without selection bias. No
  history → **shadow mode**: the decision model scores every case, takes
  over none, until enough labels exist (CRC ≥ 99 at α=1%; high-probability
  ≥ 299 at α=1%, 95%).
- **After go-live: random audit sample.** `decide()` forces ESCALATE on a
  fixed fraction ε (e.g. 5%) of traffic, chosen at random *independently of
  the model's score*, with `reason="audit"`. Those cases get the expensive
  verdict and are exchangeable with live traffic, so they are a valid CRC
  calibration set as-is — no importance weighting. Cost: ε of the savings
  (e.g. 60% → ~57%). At 10k cases/month, 5% gives 500 labels/month.
- **Organic escalations stay out of the calibration set — directly.** They are
  labelled but selected by low confidence, so mixing them in unweighted biases
  the certificate. Use them for diagnostics. (They *can* be used with
  inverse-probability weighting; see open problems.)
- **Delayed and noisy labels.** Reverts / incidents can arrive weeks later:
  certify only on cases older than a maturity window. The expensive path's
  verdict is itself fallible, so a certificate holds *relative to that
  path's verdict*; say so in contracts and docs. Incidents and reverts are
  alarms (they reveal only some misses), never an estimate of miss rate.
- **Scheduled recertify, with label-free drift as an early trigger.**
  Recertify whenever enough audit labels have matured (~150 at α=5% restored
  α in findings §10). Between runs, `drift` watches automation rate and the
  confidence distribution (PSI / KS); a trigger brings recertify forward or
  temporarily raises ε. Decided 2026-10-24 after measuring that drift misses
  concept drift — a drift `ok` never replaces a scheduled recertify.

Open problems (not solved by the plan above):

- **Window vs staleness.** Audit samples accumulate over months. After drift,
  old samples are no longer exchangeable with current traffic. A short
  window starves the certificate, a long one certifies the past. Needs a
  rolling-window rule tied to the drift monitor (drop pre-drift samples on a
  trigger).
- **Low-volume customers fill slowly.** At 1k cases/month and ε=5% that is 50
  labels/month: ~2 months to the CRC floor of 99 (α=1%), ~6 months to the
  high-probability floor of 299. Options: raise ε, accept longer cycles, or
  certify at a looser α. The mechanism suits mid/high-volume traffic best.
- **Using escalations via IPW (later optimisation).** With audit sampling every
  case's labelling probability is known (1 if escalated, ε if auto-accepted),
  so inverse-probability weighting can use all labelled cases. Not in v0.1: it
  inflates the per-case loss bound from 1 to 1/ε, and CRC's `B/(n+1)` term
  grows with it — at ε=5%, α=1% the minimum calibration size goes from 99 to
  ~2000. Revisit only if audit-only samples prove too few.

Work items:

| Item | What | When |
|---|---|---|
| ~~Audit sampling in `decide()`~~ | done 2026-10-23: `audit_rate` / `audit_seed` in the profile, `Decision.audit` | before v0.1 |
| ~~Outcome feedback format~~ | done 2026-10-24: `OutcomeFeedback`, `load_outcomes` | 0.1.x |
| ~~`decisionops recertify`~~ | done 2026-10-24: audit-only CRC, profile diff | 0.1.x |
| ~~Drift monitor~~ | done 2026-10-24: `decisionops drift`, PSI + KS | 0.1.x |
| ~~Shadow-mode guide~~ | done 2026-10-24: `docs/shadow-mode.md` + demo script | docs, 0.1.x |

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
| Decision contract | `contract.py` | Closed-set choices, auto-set, costs, severity (true-label miss weight), yes_choice; asymmetric + weighted risk for CRC/gate |
| Offline cascade | `cascade.py` | Cost-saved vs joint miss-rate curve; oracle escalate pricing; needs contract costs |
| Runtime decide | `decide.py` | `CertificationProfile` from a certify report; `decide()` → AUTO/ESCALATE; model + contract-hash guards |
| Reports | `report/` | Rich console + JSON + self-contained HTML (reliability, risk-coverage, cascade cost curve) |

### CLI commands

| Command | Status |
|---|---|
| `decisionops eval` | done — same-set smoke; `--provider static\|laya\|jev\|http` |
| `decisionops gate` | done — cal/test split; temperature + isotonic when `--calibrate`; `--provider static\|laya\|jev\|http` |
| `decisionops compare` | done — shared split, CRC ranking, empirical table retained with `!`; offline JSONL only |
| `decisionops certify` | done — CRC thresholds; `--provider static\|laya\|jev\|http`; `--profile` writes a runtime profile |
| `decisionops cascade` | done — cost-saved vs miss-rate; requires `--contract` with costs |
| `decisionops decide` | done — route a predictions JSONL with a profile; offline smoke |
| `decisionops recertify` | done — audit-only CRC from decisions×outcomes×predictions; profile diff |

Provider-specific flags (`--model-dir`, `--no-builtin-temperature`, `--jev-model`,
`--jev-base-url`, `--http-url`, `--http-token`) are rejected when combined with a
different `--provider`.

### Providers

| Provider | Location | Status |
|---|---|---|
| Static file | `adapters/base.py` | done — offline predictions, used for CI |
| Laya | `adapters/laya.py` | done — local inference; `--model-dir` for a fine-tuned checkpoint, `--no-builtin-temperature` |
| TypeSafe Jev | `adapters/jev.py` | done — hosted Choice + Noul via `typesafe-sdk==0.7.2` (`--extra jev`); `--jev-model` (incl. fine-tuned ids), `--jev-base-url`; export via `scripts/export_jev_predictions.py`. Live-verified 2026-09-27 against `jev-1.13.0` (Choice + Noul). BANKING77 n=500 compare in [findings.md](findings.md) §8 |
| Generic HTTP | `adapters/http.py` | done — POST `{id,state,question,choices}` (never `label`); `--http-url` required, `--http-token` / `DECISIONOPS_HTTP_TOKEN` optional Bearer; stdlib `urllib` |
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

- **175 tests passing**, 4 skipped (live Jev smoke without a key; 3 Laya tests
  without torch). 87% coverage with `--extra dev --extra jev` (CI's setup),
  ruff clean on `src tests` (four pre-existing errors in older `scripts/`,
  which CI does not lint).

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
