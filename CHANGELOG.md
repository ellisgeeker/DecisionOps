# Changelog

## Unreleased

### Added

- `certify --risk selective` (`run_certification(risk="selective")`,
  `certify_selective_threshold`): certifies the error rate **among**
  auto-accepted cases with Learn-Then-Test, high-probability only, 0-1 loss.
  Joint CRC stays the default. `recertify` keeps a selective profile selective.
- findings §11: joint CRC at α=5% leaves ~10.6% error among automated cases;
  the selective certificate exceeds α in 3–8% of draws against a 10% budget, at
  the cost of much lower coverage.

## 0.1.1 — 2026-10-24

Closes the label loop after go-live: audit labels feed recertification, and a
label-free monitor watches between runs.

### Added

- Outcome feedback JSONL (`OutcomeFeedback`: id, label, label_source, matured_at).
- `decisionops recertify` — CRC on `audit=True` rows only; diffs against the live
  profile; optional `--as-of` / `--since` maturity window; writes a replacement
  profile with `split.protocol=audit_recertify`.
- `decisionops drift` — label-free monitor of certified rate and confidence
  (PSI + KS) vs a baseline decisions window; exit 0/1/2 for ok/watch/recertify_suggested.
- Library: `run_recertification`, `run_drift_monitor`, `load_outcomes`, `load_decisions`.
- Docs: `docs/shadow-mode.md` and `scripts/run_shadow_mode_demo.py` (BANKING77 walkthrough).
- `scripts/measure_label_loop.py` and findings §10: the loop measured end to end.

### Measured

On 200 simulated drifts (α=5%, closed-form true risk):

- Drift roughly doubles the old threshold's true risk (0.049 → 0.095), above α
  in every rep.
- `recertify` on ~150 audit labels restores mean risk to 0.046–0.049.
- Recertifying on organic escalations is worse than not recertifying (0.122):
  `recertify` reads `audit=True` rows only.
- `drift` cannot see concept drift (`ok` on 190/200 while risk doubled).

### Changed

- Operations guidance: **recertify on a schedule**; `drift` only brings a run
  forward. A drift `ok` is not evidence the certificate still holds, and its
  console note now says so.
- Export scripts write repo-relative paths in committed stats files.

### Known limits

- Label delay and label noise are not modelled in the loop measurement.
- Window-vs-staleness and low-volume label supply remain open (see `docs/status.md`).

## 0.1.0 — 2026-10-24

First release. DecisionOps certifies when a cheap decision model's choice can
run automatically and when it must escalate to an expensive path.

### Commands

- `eval` — same-set metrics smoke (accuracy, ECE, Brier, NLL, latency).
- `gate` — cal/test split; go / watch / no-go from a CRC certificate at
  `--verdict-alpha`, coverage bar `--min-coverage`. ECE is context only.
- `certify` — Conformal Risk Control thresholds on the joint automated error;
  `--profile` freezes one alpha into a runtime profile.
- `compare` — several models on a shared split, ranked by certified coverage.
- `cascade` — cost saved vs miss rate on held-out data (empirical, oracle-priced).
- `decide` — route a predictions file with a profile: AUTO or ESCALATE.

### Guarantees

- Default: CRC, joint automated error ≤ α **in expectation** over calibration draws.
- `--confidence-level 0.95`: high-probability (Clopper-Pearson RCPS) certificate,
  0-1 loss only.
- Decision contracts: only auto-path errors count as risk; `severity` weights a
  miss by its true label (B = max severity).

### Runtime

- `CertificationProfile` records threshold, temperature, model and contract
  fingerprint; `decide()` refuses a different model or a changed contract.
- Random audit sample (`--audit-rate`): a seeded, score-independent share of
  traffic is forced to the expensive path and marked `audit=True`, for
  unbiased recertification.

### Providers

Static JSONL, Laya (local, `--extra laya`), TypeSafe Jev (`--extra jev`),
generic HTTP (never sends the gold label; refuses redirects).

### Known limits

- No outcome feedback, `recertify` or drift monitor (added in 0.1.1).
- Conditional (selective) risk and cascade thresholds are not certified.
- Ordinal (Score) questions are not supported.
- Validated on public benchmarks only (BANKING77, CLINC OOS), not production traffic.
