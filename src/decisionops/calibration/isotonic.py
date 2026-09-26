"""Isotonic regression for post-hoc confidence calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from decisionops.schemas import DecisionCase, DecisionPrediction

_EPS = 1e-12


@dataclass(frozen=True)
class IsotonicCalibration:
    """Binary isotonic map from max-probability to empirical correctness."""

    regressor: IsotonicRegression
    n_calibration: int
    nll_before: float
    nll_after: float

    def apply(self, prediction: DecisionPrediction) -> DecisionPrediction:
        return apply_isotonic(prediction, self)


def fit_isotonic(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
) -> IsotonicCalibration:
    """
    Fit isotonic regression on calibration pairs.

    Maps confidence (max-class probability) to empirical correctness on the
    calibration split. The fitted map is then applied to full distributions
    while preserving the predicted label (see `apply_isotonic`).
    """
    if len(pairs) < 2:
        raise ValueError("need at least 2 calibration pairs to fit isotonic")

    confidences: list[float] = []
    correct: list[float] = []
    nll_before = 0.0

    for case, pred in pairs:
        confidences.append(pred.confidence)
        correct.append(1.0 if pred.prediction == case.label else 0.0)
        nll_before += -math.log(max(pred.probability_for(case.label), _EPS))

    x = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)

    # Constant labels still fit; the map collapses to the empirical mean.
    regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    regressor.fit(x, y)

    fitted = IsotonicCalibration(
        regressor=regressor,
        n_calibration=len(pairs),
        nll_before=float(nll_before / len(pairs)),
        nll_after=0.0,  # filled below after apply
    )
    calibrated = apply_isotonic_to_pairs(pairs, fitted)
    nll_after = 0.0
    for case, pred in calibrated:
        nll_after += -math.log(max(pred.probability_for(case.label), _EPS))
    nll_after /= len(calibrated)

    return IsotonicCalibration(
        regressor=regressor,
        n_calibration=len(pairs),
        nll_before=float(nll_before / len(pairs)),
        nll_after=float(nll_after),
    )


def apply_isotonic(
    prediction: DecisionPrediction,
    calibration: IsotonicCalibration,
) -> DecisionPrediction:
    """
    Recalibrate probabilities with a fitted isotonic map.

    The winning class confidence is mapped through the isotonic curve; residual
    mass is redistributed over the other classes in proportion to their current
    mass. The predicted label is preserved (confidence is clamped just enough
    to keep the argmax), matching temperature scaling's accuracy-preserving
    contrast.
    """
    probs = dict(prediction.probabilities)
    winner = prediction.prediction
    if winner not in probs:
        winner = max(probs, key=probs.get)

    old_conf = float(probs[winner])
    new_conf = float(calibration.regressor.predict([old_conf])[0])
    new_conf = min(max(new_conf, 0.0), 1.0)

    others = {lab: mass for lab, mass in probs.items() if lab != winner}
    rest = float(sum(others.values()))

    if rest <= _EPS:
        # Degenerate distribution; keep a tiny residual for numerical NLL.
        new_conf = min(max(new_conf, _EPS), 1.0 - _EPS)
        new_probs = {winner: new_conf}
        leftover = 1.0 - new_conf
        # Split leftover equally across non-winner keys if any exist in schema;
        # otherwise put it back on the winner (shouldn't happen: ≥2 classes).
        other_keys = [lab for lab in probs if lab != winner]
        if other_keys:
            share = leftover / len(other_keys)
            for lab in other_keys:
                new_probs[lab] = share
        else:
            new_probs[winner] = 1.0
        return prediction.model_copy(update={"probabilities": new_probs})

    # Preserve argmax: new_conf must stay ≥ every redistributed rival.
    max_other = max(others.values())
    min_to_win = max_other / (rest + max_other)
    new_conf = max(new_conf, min_to_win + _EPS)
    new_conf = min(new_conf, 1.0 - _EPS)

    scale = (1.0 - new_conf) / rest
    new_probs = {winner: new_conf}
    for lab, mass in others.items():
        new_probs[lab] = float(mass * scale)

    return prediction.model_copy(update={"prediction": winner, "probabilities": new_probs})


def apply_isotonic_many(
    predictions: list[DecisionPrediction],
    calibration: IsotonicCalibration,
) -> list[DecisionPrediction]:
    return [apply_isotonic(p, calibration) for p in predictions]


def apply_isotonic_to_pairs(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    calibration: IsotonicCalibration,
) -> list[tuple[DecisionCase, DecisionPrediction]]:
    return [(case, apply_isotonic(pred, calibration)) for case, pred in pairs]
