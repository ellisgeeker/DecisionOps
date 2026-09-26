"""Temperature scaling for multiclass probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from decisionops.schemas import DecisionCase, DecisionPrediction

_EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _probs_to_logits(probs: dict[str, float], labels: list[str]) -> np.ndarray:
    return np.log(np.array([max(probs.get(lab, 0.0), _EPS) for lab in labels], dtype=np.float64))


def _nll_with_temperature(
    log_temps: float,
    logits_list: list[np.ndarray],
    label_indices: list[int],
) -> float:
    t = math.exp(log_temps)
    # T > 0 always via exp parameterization
    total = 0.0
    for logits, y in zip(logits_list, label_indices, strict=True):
        p = _softmax(logits / t)
        total += -math.log(max(float(p[y]), _EPS))
    return total / len(logits_list)


@dataclass(frozen=True)
class TemperatureScaling:
    """Scalar temperature fit on a calibration split."""

    temperature: float
    n_calibration: int
    nll_before: float
    nll_after: float

    def apply(self, prediction: DecisionPrediction) -> DecisionPrediction:
        return apply_temperature(prediction, self.temperature)


def fit_temperature(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> TemperatureScaling:
    """
    Fit temperature T by minimizing NLL on calibration pairs.

    Operates on probability vectors by converting to logits with log(p), then
    applying softmax(logits / T). Prediction argmax is preserved when T > 0.
    """
    if len(pairs) < 2:
        raise ValueError("need at least 2 calibration pairs to fit temperature")

    logits_list: list[np.ndarray] = []
    label_indices: list[int] = []
    nll_before = 0.0

    for case, pred in pairs:
        labels = list(pred.probabilities.keys())
        if case.label not in labels:
            # Gold outside support: skip for fitting but rare with aligned data.
            continue
        logits = _probs_to_logits(pred.probabilities, labels)
        y = labels.index(case.label)
        logits_list.append(logits)
        label_indices.append(y)
        nll_before += -math.log(max(pred.probability_for(case.label), _EPS))

    if len(logits_list) < 2:
        raise ValueError("not enough calibration pairs with gold label in support")

    nll_before /= len(logits_list)

    result = minimize_scalar(
        lambda log_t: _nll_with_temperature(log_t, logits_list, label_indices),
        bounds=(math.log(0.05), math.log(10.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = float(math.exp(result.x))
    nll_after = float(result.fun)

    return TemperatureScaling(
        temperature=temperature,
        n_calibration=len(logits_list),
        nll_before=float(nll_before),
        nll_after=nll_after,
    )


def apply_temperature(
    prediction: DecisionPrediction,
    temperature: float,
) -> DecisionPrediction:
    """Return a copy with probabilities re-scaled by temperature T."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    labels = list(prediction.probabilities.keys())
    logits = _probs_to_logits(prediction.probabilities, labels)
    scaled = _softmax(logits / temperature)
    probs = {lab: float(scaled[i]) for i, lab in enumerate(labels)}
    # Argmax stable for T>0, but recompute for safety under float noise.
    new_pred = max(probs, key=probs.get)
    return prediction.model_copy(update={"prediction": new_pred, "probabilities": probs})


def apply_temperature_many(
    predictions: list[DecisionPrediction],
    temperature: float,
) -> list[DecisionPrediction]:
    return [apply_temperature(p, temperature) for p in predictions]


def apply_temperature_to_pairs(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    temperature: float,
) -> list[tuple[DecisionCase, DecisionPrediction]]:
    return [(case, apply_temperature(pred, temperature)) for case, pred in pairs]
