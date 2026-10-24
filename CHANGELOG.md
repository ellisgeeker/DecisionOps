# Changelog

## Unreleased

### Added

- Outcome feedback JSONL (`OutcomeFeedback`: id, label, label_source, matured_at).
- `decisionops recertify` — CRC on `audit=True` rows only; diffs against the live
  profile; optional `--as-of` / `--since` maturity window; writes a replacement
  profile with `split.protocol=audit_recertify`.
- `decisionops drift` — label-free monitor of certified rate and confidence
  (PSI + KS) vs a baseline decisions window; exit 0/1/2 for ok/watch/recertify_suggested.
- Library: `run_recertification`, `run_drift_monitor`, `load_outcomes`, `load_decisions`.
- Docs: `docs/shadow-mode.md` and `scripts/run_shadow_mode_demo.py` (BANKING77 walkthrough).

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

- Conditional (selective) risk and cascade thresholds are not certified.
- Ordinal (Score) questions are not supported.
- Validated on public benchmarks only (BANKING77, CLINC OOS), not production traffic.
