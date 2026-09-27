"""
Jev adapter tests against a fake client, so they run without a key or network.

The live smoke test at the bottom only runs when both TYPESAFE_API_KEY and
DECISIONOPS_LIVE_JEV=1 are set, so CI never spends credits by accident.
"""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest

from decisionops.schemas import DecisionCase

_HAS_SDK = importlib.util.find_spec("typesafe_sdk") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_SDK, reason="typesafe-sdk not installed (uv sync --extra jev)"
)

CASE = DecisionCase(
    id="c1",
    state="Customer was charged twice for order 4412.",
    question="What should we do?",
    choices=["refund", "reject", "review"],
    label="refund",
)


class FakeClient:
    """Records every system_one call and returns a canned Choice answer."""

    def __init__(self, probabilities, choice=None, model="jev-1.13.0"):
        self.calls = []
        self.probabilities = probabilities
        self.choice = choice or max(probabilities, key=probabilities.get)
        self.model = model

    def system_one(self, *, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        answer = SimpleNamespace(
            type="choice",
            choice=self.choice,
            confidence=0.5,
            probabilities=dict(self.probabilities),
        )
        return SimpleNamespace(model=self.model, answers={"decision": answer}, usage=None)


def _provider(client, **kwargs):
    from decisionops.adapters.jev import JevProvider

    return JevProvider(client=client, **kwargs)


def test_predict_maps_choice_answer_to_prediction():
    client = FakeClient({"refund": 0.8, "reject": 0.05, "review": 0.15})
    pred = _provider(client).predict(CASE)

    assert pred.id == "c1"
    assert pred.prediction == "refund"
    assert pred.probabilities == pytest.approx({"refund": 0.8, "reject": 0.05, "review": 0.15})
    assert pred.provider == "jev"
    assert pred.model == "jev-1.13.0"  # the model the API actually served
    assert pred.latency_ms is not None and pred.latency_ms >= 0
    # Jev's own confidence is recorded but not used: DecisionOps uses max-prob.
    assert pred.metadata["jev_confidence"] == 0.5
    assert pred.confidence == pytest.approx(0.8)


def test_question_uses_case_state_choices_and_question():
    client = FakeClient({"refund": 0.9, "reject": 0.05, "review": 0.05})
    _provider(client, model="my-finetuned-jev").predict(CASE)

    call = client.calls[0]
    assert call["state"] == CASE.state
    assert call["model"] == "my-finetuned-jev"
    question = call["questions"]["decision"]
    assert question.instructions == "What should we do?"
    assert question.criteria == {"refund": None, "reject": None, "review": None}


def test_contract_descriptions_and_instructions_reach_criteria():
    client = FakeClient({"refund": 0.9, "reject": 0.05, "review": 0.05})
    _provider(
        client,
        descriptions={"refund": "charged in error", "review": "unclear, needs a human"},
        contract_instructions="Route this ticket.",
    ).predict(CASE)

    question = client.calls[0]["questions"]["decision"]
    assert question.instructions == "Route this ticket."
    assert question.criteria == {
        "refund": "charged in error",
        "reject": None,
        "review": "unclear, needs a human",
    }


def test_explicit_instructions_override_contract():
    client = FakeClient({"refund": 0.9, "reject": 0.05, "review": 0.05})
    _provider(client, instructions="Explicit.", contract_instructions="Contract.").predict(CASE)
    assert client.calls[0]["questions"]["decision"].instructions == "Explicit."


def test_omitted_options_get_zero_mass():
    client = FakeClient({"refund": 0.7, "review": 0.3})
    pred = _provider(client).predict(CASE)
    assert set(pred.probabilities) == set(CASE.choices)
    assert pred.probabilities["reject"] == 0.0


def test_rejects_options_outside_case_choices():
    client = FakeClient({"refund": 0.5, "escalate": 0.5}, choice="refund")
    with pytest.raises(ValueError, match="outside case choices"):
        _provider(client).predict(CASE)


def test_predict_many_calls_once_per_case_in_order():
    client = FakeClient({"refund": 0.9, "reject": 0.05, "review": 0.05})
    cases = [CASE.model_copy(update={"id": f"c{i}"}) for i in range(3)]
    preds = _provider(client).predict_many(cases)
    assert [p.id for p in preds] == ["c0", "c1", "c2"]
    assert len(client.calls) == 3


def test_api_errors_name_the_case():
    from typesafe_sdk import TypeSafeError

    class Failing(FakeClient):
        def system_one(self, **kwargs):
            raise TypeSafeError("rate limited")

    with pytest.raises(RuntimeError, match="'c1'.*rate limited"):
        _provider(Failing({"refund": 1.0})).predict(CASE)


def test_missing_key_exits_with_clear_message(monkeypatch):
    from decisionops.adapters.jev import JevProvider

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        JevProvider()


def test_base_url_reaches_sdk_client(monkeypatch):
    import typesafe_sdk

    from decisionops.adapters.jev import JevProvider

    seen = {}

    def fake_client(**kwargs):
        seen.update(kwargs)
        return FakeClient({"refund": 1.0, "reject": 0.0, "review": 0.0})

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", fake_client)
    provider = JevProvider(api_key="k", base_url="https://jev.internal.example", model="ft-1")
    assert seen["base_url"] == "https://jev.internal.example"
    assert seen["model"] == "ft-1"
    assert provider.predict(CASE).metadata["base_url"] == "https://jev.internal.example"


@pytest.mark.skipif(
    not (os.environ.get("TYPESAFE_API_KEY") and os.environ.get("DECISIONOPS_LIVE_JEV") == "1"),
    reason="live Jev smoke: set TYPESAFE_API_KEY and DECISIONOPS_LIVE_JEV=1",
)
def test_live_jev_smoke():
    from decisionops.adapters.jev import JevProvider

    pred = JevProvider().predict(CASE)
    assert pred.prediction in CASE.choices
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-3
