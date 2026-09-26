"""Latency percentile aggregation."""

from __future__ import annotations

import numpy as np

from decisionops.schemas import DecisionPrediction, LatencyStats


def latency_stats(predictions: list[DecisionPrediction]) -> LatencyStats:
    values = [p.latency_ms for p in predictions if p.latency_ms is not None]
    if not values:
        return LatencyStats(count=0)
    arr = np.asarray(values, dtype=np.float64)
    return LatencyStats(
        count=int(arr.size),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
    )
