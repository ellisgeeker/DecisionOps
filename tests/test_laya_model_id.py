"""
Laya model id carries the bundled-temperature setting.

Pure logic, no torch: runs in CI, unlike tests/test_laya_adapter.py.
"""

from __future__ import annotations

import pytest

from decisionops.adapters.laya import laya_model_id
from decisionops.decide import CertificationProfile, ProfileMismatchError, decide
from decisionops.schemas import DecisionPrediction


def test_model_id_marks_disabled_builtin_temperature():
    assert laya_model_id("rl-agent") == "rl-agent"
    assert laya_model_id("rl-agent", subdir="ft1") == "rl-agent/ft1"
    off = laya_model_id("rl-agent", apply_builtin_temperature=False)
    assert off == "rl-agent+no-builtin-temperature"
    assert laya_model_id("rl-agent", subdir="ft1", apply_builtin_temperature=False) == (
        "rl-agent/ft1+no-builtin-temperature"
    )


def test_decide_refuses_profile_certified_with_the_other_temperature_setting():
    """The bug: same checkpoint, different probabilities, silently accepted."""
    on, off = laya_model_id("rl-agent"), laya_model_id("rl-agent", apply_builtin_temperature=False)
    profile = CertificationProfile(alpha=0.05, threshold=0.9, model=on, n_calibration=300)
    pred = DecisionPrediction(
        id="x", prediction="a", probabilities={"a": 0.95, "b": 0.05}, model=off
    )
    with pytest.raises(ProfileMismatchError, match="profile certifies"):
        decide(pred, profile)
    assert decide(pred.model_copy(update={"model": on}), profile).action == "auto"
