# Shadow mode: first certificate before go-live

DecisionOps certifies a **routing rule**: when a cheap decision model may run
automatically, and when traffic must take the expensive path (stronger model,
human, or multi-step cascade). The certificate needs labels that look like
**live traffic**, not only the cases the model already escalates.

This guide is the pre-go-live chapter of that loop. After go-live, see also:

- random audit sampling in `decide()` (`--audit-rate`)
- label-free `decisionops drift`
- audit-only `decisionops recertify`

BANKING77 is a **public practice pack** (77-way intent routing), not your
product scenario. Swap in your JSONL and contract when you have them; the
sequence stays the same.

---

## Why shadow mode

Once AUTO is live, cases the cheap path accepts never get the expensive
verdict. Organic labels then come only from escalations — selected by low
confidence — and **cannot** recalibrate CRC unweighted.

Before go-live you usually still send **everything** down the expensive path
(or you have full history that did). That is an unbiased label stream:

| Phase | Decision model | Expensive path | Labels |
|-------|----------------|----------------|--------|
| **Shadow** | Scores every case | Still runs on all (or history already did) | Full distribution |
| **Go-live** | AUTO when certified | Only escalate + audit sample | Audit rows for recertify |

**Shadow mode** means: score everything, take over **nothing**, until a
certificate exists. No history → stay in shadow until the floors below.

### How many labels

Rough floors for 0-1 loss (raise them if `contract.severity` sets \(B > 1\)):

| Target | CRC (in expectation) | High-probability 95% |
|--------|----------------------|----------------------|
| α = 5% | n ≥ 19 | n ≥ 59 |
| α = 1% | n ≥ 99 | n ≥ 299 |

Prefer several hundred matured labels before the first go-live cutover.
Low-volume traffic may need a higher audit rate later or a looser α.

---

## End-to-end loop (what you are proving)

```text
[shadow labels = full history]
        │
        ▼
   certify --profile --audit-rate ε
        │
        ▼
   decide   ── decisions.jsonl (AUTO / ESCALATE, audit flags)
        │
        ├─► drift(baseline vs live)     # no labels
        │
        └─► outcomes on audit=true      # expensive verdict, matured
                    │
                    ▼
              recertify → profile_next
```

---

## One-command demo (BANKING77)

Offline sklearn (or other) predictions already in tree once the pack is built:

```bash
# If dataset/predictions JSONL are missing (gitignored):
uv run python scripts/prepare_hf_intent_gates.py --pack banking77
uv run python scripts/export_multi_providers.py --pack banking77

# Full shadow → certify → decide → drift → recertify
uv run python scripts/run_shadow_mode_demo.py

# Faster smoke (800 rows)
uv run python scripts/run_shadow_mode_demo.py --limit 800 --out /tmp/shadow-demo
```

### Measured smoke (800 rows, α=5%, ε=5%, seed 42)

Run on 2026-10-24 from `examples/banking77` held-out predictions:

| Step | Result |
|------|--------|
| Certify | feasible, λ̂ ≈ 0.80, cal coverage ≈ 80% |
| Decide | ~632/800 AUTO, **36** in audit sample |
| Drift | baseline vs 2nd half of same run → `ok` (same-run split is only a smoke) |
| Recertify | 36 audit labels (+ organic rows ignored) → feasible, verdict `replace` |

Artifacts land under `examples/banking77/shadow-demo/` (local; JSONL gitignored):

- `contract.json` — demo contract: all 77 intents automatic (selective CRC)
- `profile.json` / `profile_next.json`
- `decisions.jsonl`, `outcomes.jsonl`
- `drift_report.json`, `recertify_report.json`

**Interpretation:** this proves the **mechanism** on public data. Do not treat
BANKING77 coverage or thresholds as your production risk budget.

---

## Same loop with the CLI (copy-paste)

After the demo script (or your own exports), the CLI shape is:

```bash
OUT=examples/banking77/shadow-demo   # or /tmp/shadow-demo
DATA=examples/banking77/dataset.jsonl
PRED=examples/banking77/predictions.jsonl

# 1) First certificate from full labels (shadow / backfill)
uv run decisionops certify "$DATA" -p "$PRED" \
  --contract "$OUT/contract.json" \
  --alphas 0.05 --calibrate \
  --profile "$OUT/profile.json" \
  --audit-rate 0.05 --audit-seed 42

# 2) Route the same pool as if live
uv run decisionops decide "$OUT/profile.json" -p "$PRED" \
  --contract "$OUT/contract.json" \
  -o "$OUT/decisions.jsonl"

# 3) Label-free drift: compare two windows from the same profile
#    (Production: baseline = last stable week; live = this week.)
uv run decisionops drift "$OUT/decisions_live.jsonl" \
  --baseline "$OUT/decisions_baseline.jsonl" \
  --profile "$OUT/profile.json" \
  -o "$OUT/drift_report.json"

# 4) Outcomes JSONL: expensive-path label per id (audit rows required for recertify)
#    {"id":"…","label":"…","label_source":"audit","matured_at":"2026-01-01T00:00:00"}
#    The demo script writes this from dataset labels.

# 5) Recertify on audit sample only
uv run decisionops recertify "$OUT/profile.json" \
  --decisions "$OUT/decisions.jsonl" \
  --outcomes "$OUT/outcomes.jsonl" \
  -p "$PRED" \
  --contract "$OUT/contract.json" \
  --as-of 2026-12-31 \
  --profile-out "$OUT/profile_next.json" \
  -o "$OUT/recertify_report.json"
```

Exit codes worth wiring in CI/cron:

- `drift`: `0` ok, `1` watch, `2` recertify_suggested  
- `recertify`: non-zero if infeasible and you passed `--profile-out`

---

## Building outcomes yourself

Each line is an `OutcomeFeedback`:

```json
{
  "id": "banking77-test-00000",
  "label": "card_linking",
  "label_source": "audit",
  "matured_at": "2026-10-01T00:00:00"
}
```

Rules:

1. **Recertify keeps only decisions with `audit=true`.**  
   Organic escalations may be logged (`label_source: escalation`) for debugging;
   they do not enter the calibration set unweighted.
2. **`matured_at` + `--as-of`:** drop labels still inside a revert/incident window.
3. **`label` must be a contract choice** (or profile choice set).
4. Join keys are **`id`**; ids must be unique per case (`decide` warns on duplicates).

---

## Cutover checklist (real traffic)

1. **Shadow or backfill** until n clears the floor for your α (and B).
2. `certify --profile --audit-rate 0.05` (tune ε to volume).
3. Ship `decide()` behind a flag; start with shadow still on if needed.
4. Log every `Decision` (JSONL or equivalent).
5. Store expensive-path verdicts as outcomes; mark audit rows.
6. Cron: `drift` on rolling windows → on `recertify_suggested`, raise ε and/or queue `recertify`.
7. Install `profile_next` only after review of the recertify diff (threshold, coverage, bound).

The certificate always holds **relative to the expensive path’s verdict** (or
your labelling rule). Say that in the contract description.

---

## What this is not

- Not a guarantee that BANKING77 numbers transfer to your domain.  
- Not training of the decision model (you bring Jev / Laya / your head).  
- Not the code-review showcase pack (same loop, different contract and labels).  
- Not a substitute for conditional (selective) risk certificates — still joint
  error (severity-weighted when configured).

---

## Related

- [README.md](../README.md) — product positioning and Python API  
- [status.md](status.md) — roadmap and label-strategy open problems  
- [findings.md](findings.md) — measured behaviour on public packs  
- `scripts/run_shadow_mode_demo.py` — automated BANKING77 walkthrough  
