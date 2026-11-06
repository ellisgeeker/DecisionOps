"""
TypeSafe Jev (hosted System One) decision model provider.

Three question types:

- **choice** (default): one Choice question over the case's `choices`; Jev
  returns the winning choice plus a full distribution, which maps directly onto
  DecisionPrediction. Jev's own `confidence` is recorded but not used —
  DecisionOps derives confidence from the distribution like every provider.
- **noul**: one calibrated yes/no question for binary cases. Jev returns
  p = P(yes), mapped to {yes_choice: p, other_choice: 1 - p}. This is the
  natural form for routing ("does this need a deep review?").

- **score**: one ordered-rubric question for ordinal contracts. The rubric is
  built from the contract's ``choice_order`` (low to high), so the levels are
  ours, not Jev's: level i is ``choice_order[i]``. Jev returns a probability
  per level, mapped back onto choice ids. The returned ``legend`` must echo the
  rubric we sent; a mismatch is an error, never a silent re-mapping.

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
QUESTION_TYPES = ("choice", "noul", "score")


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


def build_score_criteria(
    choice_order: list[str], descriptions: dict[str, str] | None = None
) -> list[str]:
    """Ordered rubric, one entry per level from 0; the choice id when undescribed."""
    descriptions = descriptions or {}
    return [descriptions.get(choice) or choice for choice in choice_order]


def resolve_instructions(
    case: DecisionCase, override: str | None = None, contract: str | None = None
) -> str:
    """Explicit override, then contract, then the case's own question."""
    return override or contract or case.question or "Pick the best option."


class JevProvider:
    """Hosted Jev via the official TypeSafe SDK, one call per case (Choice, Noul, Score)."""

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
        choice_order: list[str] | None = None,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"question_type must be one of {QUESTION_TYPES}, got {question_type!r}"
            )
        if question_type == "noul" and not yes_choice:
            raise ValueError("noul needs yes_choice: the choice that means 'yes'")
        if question_type == "score":
            if not choice_order or len(choice_order) < 2:
                raise ValueError(
                    "score needs choice_order (low to high, at least 2 levels) "
                    "from an ordinal contract"
                )
            if len(set(choice_order)) != len(choice_order):
                raise ValueError(f"score choice_order has duplicates: {choice_order}")
        sdk = _require_sdk()
        self._sdk = sdk
        self.question_type = question_type
        self.yes_choice = yes_choice
        self.choice_order = list(choice_order) if choice_order else None
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

    def _score_levels(self, case: DecisionCase) -> list[str]:
        """The rubric order; the case must offer exactly the ordered levels."""
        assert self.choice_order is not None
        if set(case.choices) != set(self.choice_order):
            raise ValueError(
                f"case {case.id!r}: score needs choices {self.choice_order}, "
                f"got {case.choices}"
            )
        return self.choice_order

    def _question(self, case: DecisionCase):
        instructions = resolve_instructions(case, self.instructions, self.contract_instructions)
        if self.question_type == "score":
            levels = self._score_levels(case)
            return self._sdk.Score(
                instructions=instructions,
                criteria=build_score_criteria(levels, self.descriptions),
            )
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

    def _score_distribution(
        self, case: DecisionCase, answer: Any
    ) -> tuple[str, dict[str, float]]:
        levels = self._score_levels(case)
        n = len(levels)
        sent = build_score_criteria(levels, self.descriptions)
        legend = {int(k): v for k, v in dict(getattr(answer, "legend", None) or {}).items()}
        if legend and (sorted(legend) != list(range(n)) or any(
            legend[i] != sent[i] for i in range(n)
        )):
            raise ValueError(
                f"case {case.id!r}: Jev score legend {legend} does not match the "
                f"{n}-level rubric sent {sent}; levels would be mis-mapped"
            )
        raw = {int(k): float(v) for k, v in dict(answer.probabilities).items()}
        unexpected = sorted(set(raw) - set(range(n)))
        if unexpected:
            raise ValueError(
                f"case {case.id!r}: Jev score returned levels outside 0..{n - 1}: {unexpected}"
            )
        probs = {levels[i]: raw.get(i, 0.0) for i in range(n)}
        if not any(v > 0.0 for v in probs.values()):
            raise ValueError(f"case {case.id!r}: Jev score probabilities are all zero")
        # Ties go to the higher level: over-rating is never ordinal risk.
        top = max(range(n), key=lambda i: (probs[levels[i]], i))
        return levels[top], probs

    def _to_prediction(
        self, case: DecisionCase, response: Any, elapsed_ms: float
    ) -> DecisionPrediction:
        answer = response.answers.get(QUESTION_KEY)
        if answer is None:
            raise ValueError(f"case {case.id!r}: Jev response has no {QUESTION_KEY!r} answer")
        if self.question_type == "noul":
            prediction, probs = self._noul_distribution(case, answer)
            return self._build(case, response, elapsed_ms, prediction, probs, {"noul": answer.noul})
        if self.question_type == "score":
            prediction, probs = self._score_distribution(case, answer)
            return self._build(
                case,
                response,
                elapsed_ms,
                prediction,
                probs,
                {
                    "jev_score": getattr(answer, "score", None),
                    "jev_confidence": getattr(answer, "confidence", None),
                    "choice_order": list(self.choice_order or []),
                },
            )
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
