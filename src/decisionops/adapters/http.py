"""
Generic HTTP decision-model provider.

POSTs one JSON body per case to a customer endpoint:

    {"id", "state", "question", "choices"}

and optionally ``instructions`` / ``descriptions`` from the contract. The gold
``label`` and case ``metadata`` are never sent — a certificate that saw the
label is not a certificate.

The endpoint must reply:

    {"prediction": "<choice>", "probabilities": {"<choice>": 0.9, ...}}

``id`` and ``model`` in the response are optional; a mismatched ``id`` is an
error. Authentication is ``Authorization: Bearer …`` from ``token`` or
``DECISIONOPS_HTTP_TOKEN``. Uses stdlib ``urllib``; no extra dependency.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from decisionops.__version__ import __version__
from decisionops.schemas import DecisionCase, DecisionPrediction

TOKEN_ENV = "DECISIONOPS_HTTP_TOKEN"
ALLOWED_SCHEMES = {"http", "https"}
# Explicit allow-list so a future schema field cannot leak into the wire body.
REQUEST_KEYS = ("id", "state", "question", "choices", "instructions", "descriptions")

Opener = Callable[[urllib.request.Request], bytes]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """
    Refuse redirects. urllib follows a 301/302/303 on POST as a GET and copies
    every header, so the Bearer token would go to whatever host the redirect
    names — and the decision body would be dropped anyway.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(
            req.full_url, code, f"redirect to {newurl!r} refused", headers, fp
        )


_URL_OPENER = urllib.request.build_opener(_NoRedirect)


def build_request(
    case: DecisionCase,
    *,
    instructions: str | None = None,
    descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Wire body for one case. Never includes ``label`` or ``metadata``."""
    payload: dict[str, Any] = {
        "id": case.id,
        "state": case.state,
        "question": case.question,
        "choices": list(case.choices),
    }
    if instructions:
        payload["instructions"] = instructions
    if descriptions:
        subset = {
            choice: descriptions[choice]
            for choice in case.choices
            if descriptions.get(choice)
        }
        if subset:
            payload["descriptions"] = subset
    return payload


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError(f"http provider url must be http(s) with a host, got {url!r}")
    return url


class HttpProvider:
    """POST JSON to a customer decision head, one call per case."""

    name = "http"

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        timeout_s: float = 30.0,
        instructions: str | None = None,
        descriptions: dict[str, str] | None = None,
        contract_instructions: str | None = None,
        opener: Opener | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self.url = _validate_url(url)
        self.token = token if token is not None else os.environ.get(TOKEN_ENV)
        self.timeout_s = timeout_s
        self.instructions = instructions
        self.descriptions = dict(descriptions or {})
        self.contract_instructions = contract_instructions
        self._opener = opener

    def predict(self, case: DecisionCase) -> DecisionPrediction:
        return self.predict_many([case])[0]

    def predict_many(
        self, cases: list[DecisionCase], *, progress_every: int = 0
    ) -> list[DecisionPrediction]:
        out: list[DecisionPrediction] = []
        for i, case in enumerate(cases):
            t0 = time.perf_counter()
            body = self._post(build_request(
                case,
                instructions=self.instructions or self.contract_instructions,
                descriptions=self.descriptions or None,
            ))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out.append(self._to_prediction(case, body, elapsed_ms))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  http: {i + 1}/{len(cases)} cases", flush=True)
        return out

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"decisionops/{__version__}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        leaked = [k for k in payload if k not in REQUEST_KEYS]
        if leaked:
            raise RuntimeError(f"http provider refused to send unexpected keys: {leaked}")
        if "label" in payload:
            raise RuntimeError("http provider refused to send label")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=data, headers=self._headers(), method="POST"
        )
        try:
            raw = self._read(request)
        except urllib.error.HTTPError as exc:
            snippet = exc.read()[:400].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code} from {self.url} for case {payload.get('id')!r}: {snippet}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP call failed to {self.url}: {exc.reason}") from exc
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"non-JSON response from {self.url} for case {payload.get('id')!r}"
            ) from exc
        if not isinstance(body, dict):
            raise ValueError(
                f"response from {self.url} must be a JSON object, got {type(body).__name__}"
            )
        return body

    def _read(self, request: urllib.request.Request) -> bytes:
        if self._opener is not None:
            return self._opener(request)
        with _URL_OPENER.open(request, timeout=self.timeout_s) as resp:
            return resp.read()

    def _to_prediction(
        self, case: DecisionCase, body: dict[str, Any], elapsed_ms: float
    ) -> DecisionPrediction:
        if "id" in body and str(body["id"]) != case.id:
            raise ValueError(
                f"case {case.id!r}: HTTP response id {body['id']!r} does not match"
            )
        try:
            prediction = str(body["prediction"])
            raw_probs = body["probabilities"]
        except KeyError as exc:
            raise ValueError(
                f"case {case.id!r}: HTTP response missing {exc.args[0]}"
            ) from exc
        if not isinstance(raw_probs, dict) or not raw_probs:
            raise ValueError(f"case {case.id!r}: probabilities must be a non-empty object")
        probs = {str(k): float(v) for k, v in raw_probs.items()}
        unexpected = sorted(set(probs) - set(case.choices))
        if unexpected:
            raise ValueError(
                f"case {case.id!r}: HTTP returned options outside case choices: {unexpected}"
            )
        for choice in case.choices:
            probs.setdefault(choice, 0.0)
        if prediction not in case.choices:
            raise ValueError(
                f"case {case.id!r}: HTTP prediction {prediction!r} not in case choices"
            )
        model = body.get("model")
        return DecisionPrediction(
            id=case.id,
            prediction=prediction,
            probabilities=probs,
            latency_ms=round(elapsed_ms, 3),
            provider=self.name,
            model=str(model) if model is not None else None,
            metadata={"url": self.url, "n_options": len(case.choices)},
        )
