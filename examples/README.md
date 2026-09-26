# Public HF intent packs for DecisionOps

These folders are **generated** (not hand-authored):

| Dir | Source | Fit | Export (eval) |
|-----|--------|-----|----------------|
| `banking77/` | [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) | train | test |
| `clinc_oos/` | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) `plus` | train | validation + test |

Head: sklearn `TfidfVectorizer` + `LogisticRegression` → full probability vectors.

## Build

```bash
uv sync --extra bench
uv run python scripts/prepare_hf_intent_gates.py --pack all --run-gate
```

Single pack:

```bash
uv run python scripts/prepare_hf_intent_gates.py --pack banking77 --run-gate
uv run python scripts/prepare_hf_intent_gates.py --pack clinc --run-gate
```

## Re-run gate only

```bash
uv run decisionops gate examples/banking77/dataset.jsonl \
  --predictions examples/banking77/predictions.jsonl \
  --output examples/banking77/gate

uv run decisionops gate examples/clinc_oos/dataset.jsonl \
  --predictions examples/clinc_oos/predictions.jsonl \
  --output examples/clinc_oos/gate
```

## Git note

`examples/banking77/**/*.jsonl` and `examples/clinc_oos/**/*.jsonl` are
**gitignored** (hundreds of MB). Summaries, gate/compare JSON/HTML, and the
small `toy-routing` / `gate-synthetic` packs stay in the repo. Rebuild the
large packs locally with the commands above before re-running those gates.

## Protocol notes

1. Classifier is fit **only** on the HF train split.
2. Exported JSONL is **held-out only** (no train leakage into DecisionOps pool).
3. `decisionops gate` then does a further random cal/test split on that held-out pool
   (temperature + thresholds on cal, metrics on test).
4. This is still a **baseline sklearn head**, not Laya/Jev — good enough to stress
   DecisionOps on real public labels.

## Attribution

- BANKING77: Casanueva et al.; use the mteb parquet mirror for modern `datasets`.
- CLINC OOS: Larson et al.; config `plus` includes an `oos` intent class.
