"""Calibration metrics: ECE and reliability bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decisionops.schemas import DecisionCase, DecisionPrediction


@dataclass(frozen=True)
class ReliabilityBin:
    bin_low: float
    bin_high: float
    count: int
    avg_confidence: float
    accuracy: float
    gap: float  # |avg_confidence - accuracy|


def _confidences_and_correct(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([pred.confidence for _, pred in pairs], dtype=np.float64)
    correct = np.array(
        [1.0 if pred.prediction == case.label else 0.0 for case, pred in pairs],
        dtype=np.float64,
    )
    return conf, correct


def reliability_bins(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Fixed-width confidence bins for a reliability diagram."""
    if not pairs:
        raise ValueError("cannot compute reliability bins on empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    conf, correct = _confidences_and_correct(pairs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []

    for i in range(n_bins):
        low, high = float(edges[i]), float(edges[i + 1])
        # Last bin is closed on the right so confidence=1.0 is included.
        right = conf <= high if i == n_bins - 1 else conf < high
        mask = (conf >= low) & right
        count = int(mask.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    bin_low=low,
                    bin_high=high,
                    count=0,
                    avg_confidence=float("nan"),
                    accuracy=float("nan"),
                    gap=float("nan"),
                )
            )
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        bins.append(
            ReliabilityBin(
                bin_low=low,
                bin_high=high,
                count=count,
                avg_confidence=avg_conf,
                accuracy=acc,
                gap=abs(avg_conf - acc),
            )
        )
    return bins


def expected_calibration_error(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error with fixed-width bins on max-probability confidence.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    """
    bins = reliability_bins(pairs, n_bins=n_bins)
    n = len(pairs)
    ece = 0.0
    for b in bins:
        if b.count == 0:
            continue
        ece += (b.count / n) * b.gap
    return float(ece)
