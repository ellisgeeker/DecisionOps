"""Train/calibration/test splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decisionops.schemas import DecisionCase, DecisionPrediction


@dataclass(frozen=True)
class SplitResult:
    """Aligned case/prediction pairs partitioned into calibration and test."""

    calibration: list[tuple[DecisionCase, DecisionPrediction]]
    test: list[tuple[DecisionCase, DecisionPrediction]]
    seed: int
    cal_ratio: float

    @property
    def n_calibration(self) -> int:
        return len(self.calibration)

    @property
    def n_test(self) -> int:
        return len(self.test)

    @property
    def n_total(self) -> int:
        return self.n_calibration + self.n_test


def split_pairs(
    pairs: list[tuple[DecisionCase, DecisionPrediction]],
    *,
    cal_ratio: float = 0.3,
    seed: int = 42,
) -> SplitResult:
    """
    Randomly partition pairs into calibration and test sets.

    Thresholds and temperature are fit on calibration only; metrics are reported
    on test only.
    """
    if not pairs:
        raise ValueError("cannot split empty pairs")
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError(f"cal_ratio must be in (0, 1), got {cal_ratio}")
    if len(pairs) < 4:
        raise ValueError(f"need at least 4 samples to split, got {len(pairs)}")

    n = len(pairs)
    n_cal = int(round(n * cal_ratio))
    n_cal = max(2, min(n - 2, n_cal))  # keep at least 2 in each split

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cal_idx = set(int(i) for i in order[:n_cal])

    calibration = [pairs[i] for i in range(n) if i in cal_idx]
    test = [pairs[i] for i in range(n) if i not in cal_idx]
    return SplitResult(
        calibration=calibration,
        test=test,
        seed=seed,
        cal_ratio=cal_ratio,
    )
