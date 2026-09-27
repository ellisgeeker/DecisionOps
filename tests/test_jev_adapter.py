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


# ------------------------------------------------------------------ noul

ROUTE = DecisionCase(
    id="r1",
    state="diff: rewrites the payment retry loop",
    question="Does this change need a deep review?",
    choices=["skip_deep_review", "deep_llm_review"],
    label="deep_llm_review",
)


class FakeNoulClient(FakeClient):
    def __init__(self, p_yes):
        super().__init__({"x": 1.0})
        self.p_yes = p_yes

    def system_one(self, *, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        answer = SimpleNamespace(type="noul", noul=self.p_yes)
        return SimpleNamespace(model=self.model, answers={"decision": answer}, usage=None)


def _noul(client, **kwargs):
    return _provider(client, question_type="noul", yes_choice="deep_llm_review", **kwargs)


def test_noul_maps_p_yes_onto_the_two_choices():
    pred = _noul(FakeNoulClient(0.82)).predict(ROUTE)
    assert pred.probabilities == pytest.approx(
        {"deep_llm_review": 0.82, "skip_deep_review": 0.18}
    )
    assert pred.prediction == "deep_llm_review"
    assert pred.metadata["question_type"] == "noul"
    assert pred.metadata["noul"] == 0.82

    low = _noul(FakeNoulClient(0.1)).predict(ROUTE)
    assert low.prediction == "skip_deep_review"
    assert low.confidence == pytest.approx(0.9)


def test_noul_tie_resolves_to_no():
    # A coin flip must not be read as a "yes, escalate" — nor as a confident skip.
    pred = _noul(FakeNoulClient(0.5)).predict(ROUTE)
    assert pred.prediction == "skip_deep_review"


def test_noul_sends_noul_question_with_descriptions_as_true_false():
    client = FakeNoulClient(0.7)
    _noul(
        client,
        descriptions={
            "deep_llm_review": "touches logic, interfaces or data",
            "skip_deep_review": "docs or formatting only",
        },
    ).predict(ROUTE)
    question = client.calls[0]["questions"]["decision"]
    assert question.type == "noul"
    assert question.instructions == "Does this change need a deep review?"
    # NoulCriteria is a TypedDict with fixed true/false keys.
    assert question.criteria == {
        "true": "touches logic, interfaces or data",
        "false": "docs or formatting only",
    }


def test_noul_without_descriptions_sends_no_criteria():
    client = FakeNoulClient(0.7)
    _noul(client).predict(ROUTE)
    assert client.calls[0]["questions"]["decision"].criteria is None


def test_noul_rejects_non_binary_cases_and_missing_yes_choice():
    from decisionops.adapters.jev import JevProvider

    with pytest.raises(ValueError, match="exactly 2 choices"):
        _noul(FakeNoulClient(0.7)).predict(CASE)  # CASE has three choices
    with pytest.raises(ValueError, match="yes_choice"):
        JevProvider(client=FakeNoulClient(0.7), question_type="noul")
    with pytest.raises(ValueError, match="question_type"):
        JevProvider(client=FakeNoulClient(0.7), question_type="score")


def test_noul_predictions_certify_through_asymmetric_contract():
    """End to end: Noul answers -> auto-path-only CRC certificate."""
    import numpy as np

    from decisionops.conformal.certify import run_certification
    from decisionops.contract import DecisionContract

    contract = DecisionContract(
        name="deep-review-routing",
        choices=["skip_deep_review", "deep_llm_review"],
        auto_choices=["skip_deep_review"],
        yes_choice="deep_llm_review",
    )
    rng = np.random.default_rng(0)
    cases, preds = [], []
    for i in range(600):
        # Calibrated model: P(needs review) == p_yes.
        p_yes = float(rng.uniform(0.0, 1.0))
        needs = bool(rng.random() < p_yes)
        case = ROUTE.model_copy(
            update={"id": str(i), "label": "deep_llm_review" if needs else "skip_deep_review"}
        )
        cases.append(case)
        preds.append(_noul(FakeNoulClient(p_yes)).predict(case))

    report = run_certification(cases, preds, alphas=[0.05], calibrate=False, contract=contract)
    level = report.levels[0]
    assert level.certificate.feasible
    # Only confident skips are auto-executed; escalations are never risk.
    assert 0.0 < level.test["coverage"] < 0.5
    assert report.metadata["contract"]["yes_choice"] == "deep_llm_review"


@pytest.mark.skipif(
    not (os.environ.get("TYPESAFE_API_KEY") and os.environ.get("DECISIONOPS_LIVE_JEV") == "1"),
    reason="live Jev smoke: set TYPESAFE_API_KEY and DECISIONOPS_LIVE_JEV=1",
)
def test_live_jev_smoke():
    from decisionops.adapters.jev import JevProvider

    pred = JevProvider().predict(CASE)
    assert pred.prediction in CASE.choices
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-3
