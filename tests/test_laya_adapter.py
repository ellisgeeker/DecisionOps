"""Laya adapter tests.

Skipped unless torch/transformers are installed and the checkpoint is cached,
so CI without the `laya` extra stays green.
"""

from __future__ import annotations

import importlib.util

import pytest

from decisionops.schemas import DecisionCase

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed (uv sync --extra laya)")


def _cached_snapshot():
    from huggingface_hub import snapshot_download

    from decisionops.adapters.laya import DEFAULT_REPO, SNAPSHOT_PATTERNS

    # local_files_only: never download inside tests
    return snapshot_download(
        DEFAULT_REPO, allow_patterns=SNAPSHOT_PATTERNS, local_files_only=True
    )


@pytest.fixture(scope="module")
def provider():
    try:
        path = _cached_snapshot()
    except Exception as exc:  # noqa: BLE001 - any cache miss means skip
        pytest.skip(f"Laya checkpoint not cached: {exc}")

    from decisionops.adapters.laya import LayaProvider

    return LayaProvider(model_dir=path, device="cpu")


def test_predict_returns_distribution_over_choices(provider):
    case = DecisionCase(
        id="c1",
        state="Customer was charged twice for order 4412.",
        question="What should we do?",
        choices=["refund", "reject", "review"],
        label="refund",
    )
    pred = provider.predict(case)
    assert pred.id == "c1"
    assert set(pred.probabilities) == set(case.choices)
    assert pred.prediction in case.choices
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6
    assert pred.provider == "laya"
    assert pred.latency_ms is not None and pred.latency_ms > 0


def test_builtin_temperature_flag_changes_sharpness(provider):
    from decisionops.adapters.laya import LayaProvider

    case = DecisionCase(
        id="c2",
        state="The API returns 503 errors since this morning.",
        question="Which tool should the agent use next?",
        choices=["search_code", "run_tests", "inspect_network", "ask_user"],
        label="inspect_network",
    )
    with_temp = provider.predict(case)
    raw = LayaProvider(
        model_dir=provider.snapshot, device="cpu", apply_builtin_temperature=False
    ).predict(case)
    # Same argmax ranking source, different reported sharpness.
    assert with_temp.metadata["temperature_applied"] != raw.metadata["temperature_applied"]
    assert raw.metadata["temperature_applied"] == 1.0
    # The setting is part of the model id, so a profile cannot cross over.
    assert raw.model == f"{with_temp.model}+no-builtin-temperature"


def test_batch_matches_single(provider):
    cases = [
        DecisionCase(id=str(i), state=f"ticket number {i} about a refund",
                     question="What should we do?",
                     choices=["refund", "reject", "review"], label="refund")
        for i in range(3)
    ]
    batched = provider.predict_many(cases, batch_size=3)
    assert [p.id for p in batched] == ["0", "1", "2"]
    single = provider.predict(cases[0])
    assert single.prediction == batched[0].prediction
    for key in single.probabilities:
        assert abs(single.probabilities[key] - batched[0].probabilities[key]) < 1e-4
