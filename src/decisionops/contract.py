"""
Decision contract: the closed choice set of one business scenario.

A contract says which choices a decision model may pick, which of those are
executed automatically (the cheap path), and which escalate to a slower, more
expensive path (a deep LLM review, a human). Escalating is always allowed and
never counts as risk; only an automatic action that turns out wrong does.

    {
      "name": "code-review-routing",
      "choices": ["skip_deep_review", "deep_llm_review", "human_review"],
      "auto_choices": ["skip_deep_review"],
      "costs": {"skip_deep_review": 0.0, "deep_llm_review": 0.4, "human_review": 25.0}
    }

The contract is scenario data, not code: code review, ticket triage, and agent
tool-call confirmation are all expressed with the same schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from decisionops.schemas import DecisionCase, DecisionPrediction


class DecisionContract(BaseModel):
    name: str
    description: str = ""
    choices: list[str] = Field(min_length=2)
    auto_choices: list[str] = Field(min_length=1)
    """Choices executed without escalation. Errors here are the certified risk."""

    costs: dict[str, float] = Field(default_factory=dict)
    """Optional per-choice cost of taking that path (any unit, used for reporting)."""

    @model_validator(mode="after")
    def _check_consistency(self) -> DecisionContract:
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"contract {self.name!r}: duplicate choices")
        unknown_auto = sorted(set(self.auto_choices) - set(self.choices))
        if unknown_auto:
            raise ValueError(
                f"contract {self.name!r}: auto_choices not in choices: {unknown_auto}"
            )
        unknown_cost = sorted(set(self.costs) - set(self.choices))
        if unknown_cost:
            raise ValueError(f"contract {self.name!r}: costs for unknown choices: {unknown_cost}")
        negative = sorted(k for k, v in self.costs.items() if v < 0)
        if negative:
            raise ValueError(f"contract {self.name!r}: negative costs for {negative}")
        return self

    @property
    def auto_set(self) -> frozenset[str]:
        return frozenset(self.auto_choices)

    @property
    def escalate_choices(self) -> list[str]:
        return [c for c in self.choices if c not in self.auto_set]

    def summary(self) -> dict[str, object]:
        """Compact form for report metadata."""
        return {
            "name": self.name,
            "auto_choices": list(self.auto_choices),
            "escalate_choices": self.escalate_choices,
            "costs": dict(self.costs),
        }

    def validate_data(
        self,
        cases: list[DecisionCase],
        predictions: list[DecisionPrediction] | None = None,
    ) -> None:
        """
        Reject data that does not fit the contract, listing the first offenders.

        Case choices must be a subset of the contract's choices, and every
        predicted choice must be one of them. A model that predicts outside the
        contract cannot be routed, so that is an error rather than a warning.
        """
        allowed = set(self.choices)
        problems: list[str] = []
        for case in cases:
            extra = sorted(set(case.choices) - allowed)
            if extra:
                problems.append(f"case {case.id!r}: choices outside contract {extra}")
        for pred in predictions or []:
            if pred.prediction not in allowed:
                problems.append(
                    f"prediction {pred.id!r}: {pred.prediction!r} is not a contract choice"
                )
        if problems:
            preview = "; ".join(problems[:5])
            more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
            raise ValueError(f"data does not fit contract {self.name!r}: {preview}{more}")


def load_contract(path: str | Path) -> DecisionContract:
    """Load a contract from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contract not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in contract {p}: {exc}") from exc
    return DecisionContract.model_validate(raw)


def auto_set_of(contract: DecisionContract | None) -> frozenset[str] | None:
    """None means "every choice is automatic" — the pre-contract behaviour."""
    return None if contract is None else contract.auto_set
