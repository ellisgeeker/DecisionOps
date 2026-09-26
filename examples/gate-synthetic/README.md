# Gate synthetic (support intent)

Offline synthetic **support-intent routing** dataset for DecisionOps technical gate.

- **n = 2000** (8 intents × 250)
- Probabilities from a sklearn TF-IDF + LogisticRegression head
- Deliberately sharpened to create miscalibration
- **No network** required; rebuild with:

```bash
uv run python scripts/build_gate_synthetic.py --n-per-class 250 --out examples/gate-synthetic
```

## Run the gate

```bash
uv run decisionops gate examples/gate-synthetic/dataset.jsonl \
  --predictions examples/gate-synthetic/predictions.jsonl \
  --cal-ratio 0.3 \
  --seed 42 \
  --output ./report/gate-synthetic
```

This is **not** a public academic benchmark and **not** production traffic.
It exists to stress-test split + temperature scaling + risk-coverage at ~2k scale
before you bring real labels.
