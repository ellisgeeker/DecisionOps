"""
TypeSafe Jev (hosted System One) decision model provider.

Two question types:

- **choice** (default): one Choice question over the case's `choices`; Jev
  returns the winning choice plus a full distribution, which maps directly onto
  DecisionPrediction. Jev's own `confidence` is recorded but not used —
  DecisionOps derives confidence from the distribution like every provider.
- **noul**: one calibrated yes/no question for binary cases. Jev returns
  p = P(yes), mapped to {yes_choice: p, other_choice: 1 - p}. This is the
  natural form for routing ("does this need a deep review?").

Score (an ordered 2–10 level scale) is not supported: its value is the ordering,
and treating levels as unordered classes would make every metric misleading.

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
QUESTION_TYPES = ("choice", "noul")


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
    """Hosted Jev via the official TypeSafe SDK, one call per case (Choice or Noul)."""

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
        question_type: str = "choice",
        yes_choice: str | None = None,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"question_type must be one of {QUESTION_TYPES}, got {question_type!r}"
            )
        if question_type == "noul" and not yes_choice:
            raise ValueError("noul needs yes_choice: the choice that means 'yes'")
        sdk = _require_sdk()
        self._sdk = sdk
        self.question_type = question_type
        self.yes_choice = yes_choice
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

    def _binary_labels(self, case: DecisionCase) -> tuple[str, str]:
        """(yes, no) labels for a Noul case; the case must have exactly two choices."""
        if len(case.choices) != 2:
            raise ValueError(
                f"case {case.id!r}: noul needs exactly 2 choices, got {case.choices}"
            )
        if self.yes_choice not in case.choices:
            raise ValueError(
                f"case {case.id!r}: yes_choice {self.yes_choice!r} not in {case.choices}"
            )
        no = next(c for c in case.choices if c != self.yes_choice)
        return self.yes_choice, no

    def _question(self, case: DecisionCase):
        instructions = resolve_instructions(case, self.instructions, self.contract_instructions)
        if self.question_type == "noul":
            yes, no = self._binary_labels(case)
            criteria = None
            if self.descriptions.get(yes) or self.descriptions.get(no):
                criteria = {"true": self.descriptions.get(yes), "false": self.descriptions.get(no)}
            return self._sdk.Noul(instructions=instructions, criteria=criteria)
        return self._sdk.Choice(
            instructions=instructions,
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

    def _noul_distribution(
        self, case: DecisionCase, answer: Any
    ) -> tuple[str, dict[str, float]]:
        p_yes = float(answer.noul)
        if not 0.0 <= p_yes <= 1.0:
            raise ValueError(f"case {case.id!r}: Jev noul {p_yes} outside [0, 1]")
        yes, no = self._binary_labels(case)
        probs = {yes: p_yes, no: 1.0 - p_yes}
        # Ties go to "no" so a coin-flip answer is never read as a confident "yes".
        return (yes if p_yes > 0.5 else no), probs

    def _to_prediction(
        self, case: DecisionCase, response: Any, elapsed_ms: float
    ) -> DecisionPrediction:
        answer = response.answers.get(QUESTION_KEY)
        if answer is None:
            raise ValueError(f"case {case.id!r}: Jev response has no {QUESTION_KEY!r} answer")
        if self.question_type == "noul":
            prediction, probs = self._noul_distribution(case, answer)
            return self._build(case, response, elapsed_ms, prediction, probs, {"noul": answer.noul})
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
        return self._build(
            case,
            response,
            elapsed_ms,
            answer.choice,
            probs,
            {"jev_confidence": getattr(answer, "confidence", None)},
        )

    def _build(
        self,
        case: DecisionCase,
        response: Any,
        elapsed_ms: float,
        prediction: str,
        probs: dict[str, float],
        extra: dict[str, Any],
    ) -> DecisionPrediction:
        usage = getattr(response, "usage", None)
        return DecisionPrediction(
            id=case.id,
            prediction=prediction,
            probabilities=probs,
            latency_ms=round(elapsed_ms, 3),
            provider=self.name,
            model=getattr(response, "model", None) or self.model,
            metadata={
                "question_type": self.question_type,
                "n_options": len(case.choices),
                "base_url": self.base_url,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
                **extra,
            },
        )
