# DecisionOps

> **Model confidence is not production reliability.**
>
> DecisionOps tells you when an AI decision is safe enough to automate.

Open-source reliability toolkit for **decision models** (routing, triage, choice heads, System-One style models): calibration, selective prediction, and risk-coverage — before you put a confidence threshold into production.

**Where it sits.** DecisionOps certifies the *decision* layer, not the execution layer. A cheap decision model (Jev, Laya, your own head) picks from a closed set of actions with probabilities; DecisionOps tells you when that choice can run automatically and when to escalate to the slow, expensive path — a deep LLM call (Claude, Qwen, DeepSeek) or a human. The certified claim: *at miss rate ≤ α, this many expensive calls are saved.*

Scenarios are described by a **decision contract** (`--contract`): the closed choice set, which choices run automatically, and optional per-path costs. Only errors on the automatic path count as risk; escalating a case that did not need it costs money, not safety. Code review ("does this diff need a deep LLM review?") is one example scenario — ticket triage, risk review, and agent tool-call confirmation use the same contract.

```json
{
  "name": "code-review-routing",
  "choices": ["skip_deep_review", "deep_llm_review", "human_review"],
  "auto_choices": ["skip_deep_review"],
  "costs": {"skip_deep_review": 0.0, "deep_llm_review": 0.4, "human_review": 25.0}
}
```

## What we found on real data

Measured on public benchmarks (BANKING77, CLINC OOS) with a local [Laya](https://huggingface.co/convaiinnovations/laya) checkpoint and four sklearn heads. Full protocol and caveats in [docs/findings.md](docs/findings.md).

**1. A shipped post-hoc calibration made things worse.** Laya bundles per-option-count temperatures. On 77-way BANKING77 it selects the `choice:11+` bucket, `T=0.10`, which *sharpens* instead of softening:

| Laya, BANKING77 (n=3076) | ECE |
|--------------------------|-----|
| with bundled `T=0.10` | **0.603** |
| without it (`T=1.0`) | 0.362 |
| after fitting `T≈6.9` on held-out data | **0.056** |

Vendor-supplied calibration is a starting point, not a reliability guarantee. This is the case for measuring it on *your* data.

**2. Confidence can carry almost no information.** Under that bundled temperature, a threshold of **1.000** still accepted 41.2% of traffic — and **44.7% of what it accepted was wrong**. `if confidence > 0.9` would have automated a coin flip.

**3. Good calibration does not mean safe automation.** After fitting temperature, Laya reached ECE **0.056** — close to a sklearn baseline's 0.034. Across 40 splits of the same BANKING77 pool, a 1% CRC certificate was **infeasible on 34/40**; when a threshold existed it covered ~2% of traffic. ECE and coverage are independent axes; a calibration library only shows you the first.

**4. Empirically-tuned thresholds rebound on held-out data.** Picking the threshold that hits 1% error on a calibration split is selecting for downward noise. On BANKING77 at a 1% target, **all five models exceeded it on test**. `compare` still prints those figures with `!`, but **ranks on CRC certificates** — infeasible models are excluded rather than reported as meeting the target.

Conformal Risk Control is the fix: it adds a `B/(n+1)` penalty and certifies `E[loss] ≤ α`. Validated against a closed-form optimum on a synthetic pool where the true threshold is known — `λ̂` sat above `λ*` in all six configurations (z = +0.45 to +3.60), so the rule is conservative as the theory requires.

**Scope.** The Laya numbers are one checkpoint, zero-shot, on one task shape. Its own docs describe it as a fine-tuning base, so low zero-shot accuracy is expected — we did **not** fine-tune, and cannot say how a tuned Laya behaves.

## Status

| Phase | Status |
|------:|--------|
| 0–1 Schemas, static provider, metrics, `eval` CLI | done |
| 2 Cal/test split + held-out thresholds | done |
| 3 Temperature scaling + `gate` report | done |
| Public HF intent packs (BANKING77, CLINC) | done |
| Model comparison (`compare`) | done |
| 4a Laya adapter (local System-One model) | done |
| 4a′ `--provider laya` on eval/gate/certify | done |
| 4c TypeSafe Jev provider (`--provider jev`) | done |
| 5 Conformal Risk Control (library, validated) | done |
| 5b `certify` CLI command | done |
| 5c HTML report (reliability + risk-coverage) | done |
| M0 git / Apache-2.0 / CI + 85% coverage gate | done |
| 4b OpenAI-compatible adapter (DeepSeek etc.) | next |
| 6+ Cascade / gateway | later |

See [docs/status.md](docs/status.md) for what is done, what is next, and known gaps.

## Quickstart

```bash
# requires: Python 3.12+, uv
uv sync --extra dev

# Offline (precomputed predictions)
uv run decisionops gate examples/gate-synthetic/dataset.jsonl \
  --predictions examples/gate-synthetic/predictions.jsonl \
  --output ./report/gate-synthetic

# Plan flagship: live Laya on a labeled dataset (needs `uv sync --extra laya`)
uv run decisionops eval examples/toy-routing/dataset.jsonl \
  --provider laya --limit 20 --output ./report/laya-toy
```

`gate` prints raw vs temperature-scaled **test** metrics, coverage at target risks with thresholds fit on **calibration only**, and a heuristic verdict (`go` / `watch` / `no-go`). `--output` writes `gate_report.json` and a self-contained `gate_report.html` with the reliability diagram and risk-coverage curve (no CDN).

## What problem this solves

Developers get:

```text
choice = refund
confidence = 0.91
```

and still do not know:

- Is 0.91 actually ~91% correct on *my* data?
- What threshold keeps automation error ≤ 1%?
- How much traffic can I auto-handle at that risk?

DecisionOps answers those offline first.

## Dataset format (JSONL)

```json
{"id":"case-001","state":"Customer was charged twice.","question":"What should we do?","choices":["refund","reject","review"],"label":"refund"}
```

## Prediction format (JSONL)

```json
{"id":"case-001","prediction":"refund","probabilities":{"refund":0.91,"reject":0.04,"review":0.05},"latency_ms":24.1,"provider":"static","model":"demo"}
```

## Examples

| Path | n | Purpose |
|------|--:|---------|
| `examples/toy-routing/` | 120 | tiny smoke test |
| `examples/gate-synthetic/` | 2000 | larger synthetic gate |
| `examples/banking77/` | ~3k held-out | public BANKING77 + sklearn probs |
| `examples/clinc_oos/` | ~8.6k held-out | public CLINC OOS plus + sklearn probs |

```bash
# synthetic (offline)
uv run python scripts/build_gate_synthetic.py --n-per-class 250 --out examples/gate-synthetic

# public HF intent packs (network once)
uv sync --extra bench
uv run python scripts/prepare_hf_intent_gates.py --pack all --run-gate
```

See `examples/README.md` (fit on train only, export held-out only).

## Model comparison

Score several models on one dataset under an **identical** cal/test split:

```bash
# generate 4 sklearn heads for a pack
uv run python scripts/export_multi_providers.py --pack banking77 --run-compare

# or compare existing prediction files
uv run decisionops compare examples/banking77/dataset.jsonl \
  -p word_lr=examples/banking77/providers/tfidf_word_lr.jsonl \
  -p char_lr=examples/banking77/providers/tfidf_char_lr.jsonl \
  -p svm_sig=examples/banking77/providers/tfidf_svm_sigmoid.jsonl \
  -p cnb=examples/banking77/providers/tfidf_cnb.jsonl \
  --output examples/banking77/compare
```

Ranking rule: **most coverage among models whose test selective risk actually held the target.** Models that broke the target are marked `!` and excluded from that ranking, so a high-accuracy but miscalibrated model cannot win by overclaiming.

## Laya provider (local System-One model)

[Laya](https://huggingface.co/convaiinnovations/laya) is an open decision model that
returns a typed probability distribution over the choices you pass in — no fine-tuning
needed to run it, options are supplied per request.

```bash
uv sync --extra laya   # torch + transformers + safetensors (~1GB)

# First run downloads ~850MB of weights into the HF cache
uv run python scripts/export_laya_predictions.py --pack banking77 --limit 800

# Same, ignoring Laya's shipped post-hoc temperature
uv run python scripts/export_laya_predictions.py --pack banking77 --limit 800 \
  --no-builtin-temperature
```

Notes:

- Runs on `mps` / `cuda` / `cpu` (auto-detected); ~100ms per case on Apple Silicon.
- Laya ships per-option-count temperatures in `rl_agent_config.json`. For 77-way
  BANKING77 it selects the `choice:11+` bucket (`T≈0.10`), which *sharpens* rather than
  softens — `--no-builtin-temperature` lets you measure that effect.
- The published checkpoint is a base model intended for fine-tuning, so zero-shot
  accuracy on a 77-class benchmark is low. It is still useful for calibration study.
- **Fine-tuned Laya:** point `--model-dir` at your checkpoint. It must keep the upstream
  layout (`rl_common.py`, `rl_agent_config.json`, `encoder/`, `model.safetensors`,
  `tokenizer/`); merge LoRA weights first. Pass `--no-builtin-temperature` — the shipped
  temperature table is usually stale after fine-tuning, and DecisionOps fits its own.

```bash
uv run decisionops gate data.jsonl --provider laya \
  --model-dir ./my-finetuned-laya --no-builtin-temperature
```

## Jev provider (hosted TypeSafe System One)

[Jev](https://docs.typesafe.ai/introduction/quickstart) is TypeSafe's hosted decision
model. Each case becomes one Choice question over the case's `choices`; the returned
distribution maps directly onto a DecisionOps prediction.

```bash
uv sync --extra jev          # typesafe-sdk
export TYPESAFE_API_KEY=...

uv run decisionops eval data.jsonl --provider jev --limit 20
uv run decisionops gate data.jsonl --provider jev --contract contract.json

# A fine-tuned model id, or a private deployment
uv run decisionops gate data.jsonl --provider jev \
  --jev-model my-finetuned-jev --jev-base-url https://jev.internal.example

# compare is offline, so export first
uv run python scripts/export_jev_predictions.py --pack toy --limit 20
```

Notes:

- One API call per case — start with `--limit` to check cost and latency.
- Jev's own `confidence` field is recorded in metadata but not used; DecisionOps derives
  confidence from the distribution, like every provider.
- Only the **Choice** primitive is supported. Score (an ordered scale) and Noul (a yes/no
  probability) do not map onto a case's own choice set, so they are out of scope.

### Choice descriptions are part of the model input

A contract can carry per-choice `descriptions` (sent as Jev `criteria`, and rendered as
`choice: text` for Laya) and shared `instructions`:

```json
{
  "name": "code-review-routing",
  "choices": ["skip_deep_review", "deep_llm_review", "human_review"],
  "auto_choices": ["skip_deep_review"],
  "instructions": "Does this change need a deep review?",
  "descriptions": {
    "skip_deep_review": "Docs, comments, tests or formatting only; no logic change",
    "deep_llm_review": "Changes business logic, an interface or a data structure",
    "human_review": "Touches auth, payments, data deletion or security-sensitive code"
  }
}
```

A certificate only holds for the input the model saw. Certify with the same descriptions
you use in production, give a fine-tuned model the descriptions it was trained with, and
recertify whenever they change. Reports record them under `metadata.contract`.

## Gate protocol

1. Join dataset ↔ predictions on `id`
2. Random split → **calibration** / **test** (`--cal-ratio`, `--seed`)
3. Fit **temperature scaling** on calibration (optional `--no-calibrate`)
4. Choose risk thresholds on calibration
5. Score Accuracy / ECE / Brier / NLL and threshold coverage on **test only**
6. Write `gate_report.json` and `gate_report.html` (reliability diagram + risk-coverage curve)

Gate thresholds are **empirical** — tuned on calibration, so they rebound on
held-out data. For thresholds that carry a finite-sample guarantee, use
`decisionops certify`.

## Certification protocol

```bash
uv run decisionops certify examples/banking77/dataset.jsonl \
  --predictions examples/banking77/providers/tfidf_char_lr.jsonl \
  --alphas 0.01,0.05,0.10 \
  --output report/certify-char_lr
```

1. Join, then split three ways: **temperature fit** / **CRC calibration** / **test**
   (`--temp-ratio` carves the first out of the calibration pool, so temperature is
   never fit on the data CRC calibrates against)
2. Certify `λ̂` per target `α` on the CRC set
3. Measure coverage and joint error on **test only**
4. Write `certification_report.json`

What is certified is the **joint automated error rate** — the share of *all*
traffic auto-accepted *and* wrong — bounded by `α` in expectation. Not the
conditional selective risk `gate` reports. Levels where the calibration set is too
small (`α ≥ B/(n+1)` fails) or where no threshold holds `α` are reported
**INFEASIBLE** rather than given a number.

## Comparison protocol

```bash
uv run decisionops compare examples/banking77/dataset.jsonl \
  -p char_lr=examples/banking77/providers/tfidf_char_lr.jsonl \
  -p svm_sig=examples/banking77/providers/tfidf_svm_sigmoid.jsonl \
  -p laya=examples/banking77/providers/laya.jsonl
```

Same shared split as `gate`, plus a CRC certificate per model. Rankings use
**certified test coverage among models with a feasible certificate**. A single
test split landing above `α` does not void the certificate (the bound is on
`E[loss]`). Empirical coverage stays in a second table, marked `!` when it
rebounds — that table does not pick the winner.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests

# Optional local coverage (CI enforces --cov-fail-under=85)
uv run pytest --cov=decisionops --cov-report=term-missing:skip-covered
```

CI runs on Python 3.12 and 3.13 (`.github/workflows/ci.yml`).

## Docs

- `decisionops-plan.md` — product plan
- `decisionops-foundations.md` — theory notes

## License

Apache License 2.0 — see [LICENSE](LICENSE).
