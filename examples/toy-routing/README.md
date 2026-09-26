# Toy agent-routing benchmark

Synthetic overconfident decision model for DecisionOps Phase 0/1.

```bash
decisionops eval examples/toy-routing/dataset.jsonl \
  --predictions examples/toy-routing/predictions.jsonl \
  --output ./report
```

This dataset is **not** a real production benchmark. It exists so you can
exercise metrics + risk-coverage before wiring Laya.
