"""
HTTP provider tests against an injected opener — no network, no extra deps.

The gold label must never leave the process. The request body is the contract
the customer's endpoint sees, so these tests pin that shape.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from urllib.request import Request

import pytest

from decisionops.adapters.http import TOKEN_ENV, HttpProvider, build_request
from decisionops.schemas import DecisionCase

CASE = DecisionCase(
    id="c1",
    state="Customer was charged twice for order 4412.",
    question="What should we do?",
    choices=["refund", "reject", "review"],
    label="refund",
    metadata={"ticket": "secret"},
)


class FakeOpener:
    """Records the Request and returns a canned JSON body."""

    def __init__(self, body: dict, *, error: Exception | None = None):
        self.body = body
        self.error = error
        self.requests: list[Request] = []

    def __call__(self, request: Request) -> bytes:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return json.dumps(self.body).encode("utf-8")


def _provider(opener, **kwargs) -> HttpProvider:
    return HttpProvider("https://model.example/v1/predict", opener=opener, **kwargs)


def test_build_request_never_includes_label_or_metadata():
    payload = build_request(CASE, instructions="Pick.", descriptions={"refund": "give money back"})
    assert set(payload) == {"id", "state", "question", "choices", "instructions", "descriptions"}
    assert "label" not in payload
    assert "metadata" not in payload
    assert payload["choices"] == ["refund", "reject", "review"]
    assert payload["descriptions"] == {"refund": "give money back"}


def test_predict_maps_response_and_records_latency():
    opener = FakeOpener(
        {
            "id": "c1",
            "prediction": "refund",
            "probabilities": {"refund": 0.8, "reject": 0.05, "review": 0.15},
            "model": "head-v2",
        }
    )
    pred = _provider(opener, token="sekrit").predict(CASE)

    assert pred.id == "c1"
    assert pred.prediction == "refund"
    assert pred.probabilities == pytest.approx({"refund": 0.8, "reject": 0.05, "review": 0.15})
    assert pred.provider == "http"
    assert pred.model == "head-v2"
    assert pred.latency_ms is not None and pred.latency_ms >= 0
    assert pred.metadata["url"] == "https://model.example/v1/predict"

    req = opener.requests[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://model.example/v1/predict"
    assert req.get_header("Authorization") == "Bearer sekrit"
    body = json.loads(req.data.decode("utf-8"))
    assert "label" not in body
    assert "metadata" not in body
    assert "secret" not in req.data.decode("utf-8")
    assert body["id"] == "c1"
    assert body["choices"] == CASE.choices


def test_token_falls_back_to_env(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "from-env")
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05}}
    )
    _provider(opener).predict(CASE)
    assert opener.requests[0].get_header("Authorization") == "Bearer from-env"


def test_missing_token_sends_no_authorization(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05}}
    )
    _provider(opener).predict(CASE)
    assert opener.requests[0].get_header("Authorization") is None


def test_rejects_non_http_url():
    with pytest.raises(ValueError, match="http"):
        HttpProvider("file:///tmp/model")
    with pytest.raises(ValueError, match="http"):
        HttpProvider("not-a-url")


def test_omitted_choice_gets_zero_mass():
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.7, "reject": 0.3}}
    )
    pred = _provider(opener).predict(CASE)
    assert pred.probabilities["review"] == pytest.approx(0.0)


def test_unexpected_choice_is_an_error():
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.5, "other": 0.5}}
    )
    with pytest.raises(ValueError, match="outside case choices"):
        _provider(opener).predict(CASE)


def test_mismatched_response_id_is_an_error():
    opener = FakeOpener(
        {
            "id": "someone-else",
            "prediction": "refund",
            "probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05},
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        _provider(opener).predict(CASE)


def test_missing_prediction_field_is_an_error():
    opener = FakeOpener({"probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05}})
    with pytest.raises(ValueError, match="missing prediction"):
        _provider(opener).predict(CASE)


def test_contract_fields_go_on_the_wire():
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05}}
    )
    _provider(
        opener,
        contract_instructions="Pick the best option.",
        descriptions={"refund": "give money back", "ignore_me": "not a choice"},
    ).predict(CASE)
    body = json.loads(opener.requests[0].data.decode("utf-8"))
    assert body["instructions"] == "Pick the best option."
    assert body["descriptions"] == {"refund": "give money back"}
    assert "ignore_me" not in body["descriptions"]


def test_explicit_instructions_override_contract():
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05}}
    )
    _provider(
        opener, instructions="Explicit.", contract_instructions="Contract."
    ).predict(CASE)
    body = json.loads(opener.requests[0].data.decode("utf-8"))
    assert body["instructions"] == "Explicit."


def test_http_error_becomes_runtime_error():
    err = urllib.error.HTTPError(
        "https://model.example/v1/predict",
        500,
        "boom",
        hdrs={},
        fp=BytesIO(b"upstream exploded"),
    )
    opener = FakeOpener({}, error=err)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        _provider(opener).predict(CASE)


def test_non_json_response_is_an_error():
    class Bad:
        def __call__(self, request: Request) -> bytes:
            return b"not-json"

    with pytest.raises(ValueError, match="non-JSON"):
        _provider(Bad()).predict(CASE)


def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        HttpProvider("https://model.example/v1/predict", timeout_s=0)


def test_predict_many_preserves_order():
    opener = FakeOpener(
        {"prediction": "refund", "probabilities": {"refund": 0.9, "reject": 0.05, "review": 0.05}}
    )
    other = CASE.model_copy(update={"id": "c2"})
    preds = _provider(opener).predict_many([CASE, other])
    assert [p.id for p in preds] == ["c1", "c2"]
    assert len(opener.requests) == 2


def test_nan_probability_is_an_error():
    # json.loads accepts NaN; it must not reach confidence as NaN.
    class NanBody:
        def __call__(self, request: Request) -> bytes:
            return b'{"prediction": "refund", "probabilities": {"refund": NaN, "reject": 0.1}}'

    with pytest.raises(ValueError, match="finite"):
        _provider(NanBody()).predict(CASE)


def test_redirect_is_refused_and_token_not_forwarded():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = HttpProvider(
            f"http://127.0.0.1:{server.server_port}/predict", token="sekrit", timeout_s=5
        )
        with pytest.raises(RuntimeError, match="HTTP 302"):
            provider.predict(CASE)
    finally:
        server.shutdown()
        server.server_close()
    assert seen == []
