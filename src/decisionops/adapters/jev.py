"""
TypeSafe Jev (hosted System One) decision model provider.

Each DecisionCase becomes one Choice question over the case's `choices`; Jev
returns the winning choice plus a full probability distribution, which maps
directly onto DecisionPrediction. Jev's own `confidence` field is ignored —
DecisionOps derives confidence from the distribution like every other provider.

Install extras first:

    uv sync --extra jev

Authentication uses TYPESAFE_API_KEY unless `api_key` is passed. `model` also
accepts a customer's fine-tuned model id, and `base_url` a private deployment.
"""

from __future__ import annotations

import time
from typing import Any

from decisionops.schemas import DecisionCase, DecisionPrediction

DEFAULT_MODEL = "jev-latest"
QUESTION_KEY = "decision"


def _require_sdk():
    try:
        import typesafe_sdk
    except ImportError as exc:
        raise SystemExit(
            "Jev provider needs the TypeSafe SDK.\nInstall with:  uv sync --extra jev"
        ) from exc
    return typesafe_sdk


def build_criteria(
    choices: list[str], descriptions: dict[str, str] | None = None
) -> dict[str, str | None]:
    """Per-choice rubric text; missing descriptions stay None (label only)."""
    descriptions = descriptions or {}
    return {choice: descriptions.get(choice) for choice in choices}


def resolve_instructions(
    case: DecisionCase, override: str | None = None, contract: str | None = None
) -> str:
    """Explicit override, then contract, then the case's own question."""
    return override or contract or case.question or "Pick the best option."


class JevProvider:
    """Hosted Jev via the official TypeSafe SDK, one Choice call per case."""

    name = "jev"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        instructions: str | None = None,
        descriptions: dict[str, str] | None = None,
        contract_instructions: str | None = None,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        sdk = _require_sdk()
        self._sdk = sdk
        self.model = model
        self.base_url = base_url
        self.instructions = instructions
        self.descriptions = dict(descriptions or {})
        self.contract_instructions = contract_instructions
        if client is not None:
            # Injected for tests; skips key checks so no network is touched.
            self._client = client
            return
        try:
            self._client = sdk.TypeSafeClient(
                api_key=api_key, model=model, base_url=base_url, timeout=timeout_s
            )
        except sdk.TypeSafeError as exc:
            raise SystemExit(f"Jev provider: {exc}") from exc

    def _question(self, case: DecisionCase):
        return self._sdk.Choice(
            instructions=resolve_instructions(
                case, self.instructions, self.contract_instructions
            ),
            criteria=build_criteria(case.choices, self.descriptions),
        )

    def predict(self, case: DecisionCase) -> DecisionPrediction:
        return self.predict_many([case])[0]

    def predict_many(
        self, cases: list[DecisionCase], *, progress_every: int = 0
    ) -> list[DecisionPrediction]:
        out: list[DecisionPrediction] = []
        for i, case in enumerate(cases):
            t0 = time.perf_counter()
            try:
                response = self._client.system_one(
                    state=case.state,
                    questions={QUESTION_KEY: self._question(case)},
                    model=self.model,
                )
            except self._sdk.TypeSafeError as exc:
                raise RuntimeError(f"Jev call failed for case {case.id!r}: {exc}") from exc
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out.append(self._to_prediction(case, response, elapsed_ms))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  jev: {i + 1}/{len(cases)} cases", flush=True)
        return out

    def _to_prediction(
        self, case: DecisionCase, response: Any, elapsed_ms: float
    ) -> DecisionPrediction:
        answer = response.answers.get(QUESTION_KEY)
        if answer is None:
            raise ValueError(f"case {case.id!r}: Jev response has no {QUESTION_KEY!r} answer")
        probs = {str(k): float(v) for k, v in dict(answer.probabilities).items()}
        unexpected = sorted(set(probs) - set(case.choices))
        if unexpected:
            raise ValueError(
                f"case {case.id!r}: Jev returned options outside case choices: {unexpected}"
            )
        # Options Jev omitted carry zero mass, so the distribution stays over case.choices.
        for choice in case.choices:
            probs.setdefault(choice, 0.0)
        if answer.choice not in case.choices:
            raise ValueError(f"case {case.id!r}: Jev choice {answer.choice!r} not in case choices")

        usage = getattr(response, "usage", None)
        return DecisionPrediction(
            id=case.id,
            prediction=answer.choice,
            probabilities=probs,
            latency_ms=round(elapsed_ms, 3),
            provider=self.name,
            model=getattr(response, "model", None) or self.model,
            metadata={
                "n_options": len(case.choices),
                "jev_confidence": getattr(answer, "confidence", None),
                "base_url": self.base_url,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            },
        )
