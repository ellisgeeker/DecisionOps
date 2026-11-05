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
      "costs": {"skip_deep_review": 0.0, "deep_llm_review": 0.4, "human_review": 25.0},
      "severity": {"deep_llm_review": 1.0, "human_review": 5.0}
    }

`costs` price taking a path (cascade reporting). `severity` weights an
auto-path miss by its *true* label: auto-skipping a case that needed a human
can cost more than auto-skipping one that needed an LLM. Omitted labels
default to 1.0; an empty map is plain 0-1 loss.

**Ordinal levels** (`choice_order`): when choices are ordered (Jev Score, severity
tiers), plain equality loss is wrong. With `choice_order` low→high, an automatic
miss is only the **under-rate** gap `rank(true) - rank(pred) > under_rate_tolerance`
(default 1: off by one level is allowed; off by two or more is an error).
Over-rating never counts as certificate risk (it costs money, not safety).
Ordinal and `severity` are mutually exclusive.

The contract is scenario data, not code: code review, ticket triage, and agent
tool-call confirmation are all expressed with the same schema.
"""

from __future__ import annotations

import hashlib
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

    severity: dict[str, float] = Field(default_factory=dict)
    """
    Optional per-true-label weight of an automatic miss.

    When the cheap path predicts wrong and the true label is `y`, the CRC loss
    is `severity.get(y, 1.0)` instead of 1. Keys must be contract choices;
    values must be > 0. Empty means every miss weighs 1 (0-1 loss). Part of the
    certificate: changing severity means recertify.
    """

    descriptions: dict[str, str] = Field(default_factory=dict)
    """
    Optional per-choice rubric text passed to the decision model as criteria.

    Part of the model's input: certify with the same descriptions the model sees
    in production, and recertify whenever they change. A fine-tuned model should
    get the descriptions it was trained with.
    """

    instructions: str | None = None
    """Optional shared question text; overrides each case's own `question`."""

    yes_choice: str | None = None
    """
    For binary (yes/no) decisions: the choice that means "yes".

    Needed to ask a model a yes/no question such as Jev Noul ("does this need a
    deep review?"), whose answer is P(yes). Only valid with exactly two choices.
    """

    choice_order: list[str] | None = None
    """
    Ordered choice labels, **low → high** (e.g. skip < shallow < deep < human).

    When set, CRC / gate loss uses ordinal under-rating instead of equality.
    Must list every contract choice exactly once. Part of the certificate.
    """

    under_rate_tolerance: int = Field(default=1, ge=0)
    """
    Largest under-rate gap that is still not an error.

    Error iff `rank(true) - rank(pred) > under_rate_tolerance`. Default 1 means
    missing by one level is allowed; missing by two or more is a miss. Only
    used when `choice_order` is set, and must be below `len(choice_order) - 1`
    so some gap still counts. Over-rating is never a miss.
    """

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
        unknown_sev = sorted(set(self.severity) - set(self.choices))
        if unknown_sev:
            raise ValueError(
                f"contract {self.name!r}: severity for unknown choices: {unknown_sev}"
            )
        nonpositive = sorted(k for k, v in self.severity.items() if v <= 0)
        if nonpositive:
            raise ValueError(
                f"contract {self.name!r}: severity must be > 0 for {nonpositive}"
            )
        unknown_desc = sorted(set(self.descriptions) - set(self.choices))
        if unknown_desc:
            raise ValueError(
                f"contract {self.name!r}: descriptions for unknown choices: {unknown_desc}"
            )
        if self.yes_choice is not None:
            if len(self.choices) != 2:
                raise ValueError(
                    f"contract {self.name!r}: yes_choice needs exactly 2 choices, "
                    f"got {len(self.choices)}"
                )
            if self.yes_choice not in self.choices:
                raise ValueError(
                    f"contract {self.name!r}: yes_choice {self.yes_choice!r} not in choices"
                )
        if self.choice_order is not None:
            if len(self.choice_order) != len(self.choices):
                raise ValueError(
                    f"contract {self.name!r}: choice_order must list every choice once "
                    f"(got {len(self.choice_order)} labels, {len(self.choices)} choices)"
                )
            if len(set(self.choice_order)) != len(self.choice_order):
                raise ValueError(f"contract {self.name!r}: choice_order has duplicates")
            missing = sorted(set(self.choices) - set(self.choice_order))
            extra = sorted(set(self.choice_order) - set(self.choices))
            if missing or extra:
                raise ValueError(
                    f"contract {self.name!r}: choice_order must match choices "
                    f"(missing={missing}, extra={extra})"
                )
            if self.severity:
                raise ValueError(
                    f"contract {self.name!r}: choice_order (ordinal under-rate loss) "
                    "cannot be combined with severity; use one loss definition"
                )
            # The largest possible gap is len-1. A tolerance >= that makes every
            # miss free, so the certificate would bound nothing.
            max_gap = len(self.choice_order) - 1
            if self.under_rate_tolerance >= max_gap:
                raise ValueError(
                    f"contract {self.name!r}: under_rate_tolerance="
                    f"{self.under_rate_tolerance} must be < {max_gap} "
                    f"(the largest gap on {len(self.choice_order)} levels); otherwise "
                    "no under-rating is ever an error"
                )
        return self

    @property
    def auto_set(self) -> frozenset[str]:
        return frozenset(self.auto_choices)

    @property
    def escalate_choices(self) -> list[str]:
        return [c for c in self.choices if c not in self.auto_set]

    @property
    def is_ordinal(self) -> bool:
        return self.choice_order is not None

    def rank_of(self, choice: str) -> int:
        """0-based rank in `choice_order` (low → high)."""
        if self.choice_order is None:
            raise ValueError(f"contract {self.name!r} has no choice_order")
        try:
            return self.choice_order.index(choice)
        except ValueError as exc:
            raise ValueError(
                f"contract {self.name!r}: {choice!r} not in choice_order"
            ) from exc

    def miss_severity(self, label: str) -> float:
        """Loss mass of one automatic miss whose true label is `label`."""
        return float(self.severity.get(label, 1.0))

    def miss_loss(self, prediction: str, label: str) -> float:
        """
        Per-example loss mass when the automatic path predicts `prediction`
        and the true label is `label` (eligibility is applied elsewhere).

        Ordinal: 1 iff under-rate gap exceeds `under_rate_tolerance`, else 0.
        Else: 0 on exact match, else `miss_severity(label)`.
        """
        if self.choice_order is not None:
            gap = self.rank_of(label) - self.rank_of(prediction)
            return 1.0 if gap > self.under_rate_tolerance else 0.0
        if prediction == label:
            return 0.0
        return self.miss_severity(label)

    def pair_loss(self, case: DecisionCase, prediction: DecisionPrediction) -> float:
        """`miss_loss` on a (case, prediction) pair — for CRC `loss_of` callbacks."""
        return self.miss_loss(prediction.prediction, case.label)

    def loss_bound(self) -> float:
        """
        Upper bound B on per-example CRC loss.

        Ordinal under-rate loss is 0-1 (B = 1). With severity, B is the max
        weight over the choice set (missing keys count as 1). Empty severity
        without ordinal is plain 0-1 loss, so B = 1.
        """
        if self.choice_order is not None:
            return 1.0
        if not self.severity:
            return 1.0
        return max(self.miss_severity(choice) for choice in self.choices)

    def summary(self) -> dict[str, object]:
        """Compact form for report metadata."""
        out: dict[str, object] = {
            "name": self.name,
            "choices": list(self.choices),
            "auto_choices": list(self.auto_choices),
            "escalate_choices": self.escalate_choices,
            "costs": dict(self.costs),
            "severity": dict(self.severity),
            # Model input that was certified; a change here means recertify.
            "descriptions": dict(self.descriptions),
            "instructions": self.instructions,
            "yes_choice": self.yes_choice,
        }
        if self.choice_order is not None:
            out["choice_order"] = list(self.choice_order)
            out["under_rate_tolerance"] = self.under_rate_tolerance
        return out


    def fingerprint(self) -> str:
        """
        SHA-256 over every field a certificate depends on.

        Covers the choice set, the auto/escalate split, miss severities, ordinal
        order / under-rate tolerance, and the model input (descriptions,
        instructions, yes_choice). Costs, name and description are excluded:
        they price or label decisions but change neither what the model sees nor
        which errors count as risk.
        """
        payload = {
            "choices": list(self.choices),
            "auto_choices": sorted(self.auto_choices),
            "severity": dict(sorted(self.severity.items())),
            "descriptions": dict(sorted(self.descriptions.items())),
            "instructions": self.instructions,
            "yes_choice": self.yes_choice,
            "choice_order": list(self.choice_order) if self.choice_order else None,
            "under_rate_tolerance": (
                self.under_rate_tolerance if self.choice_order is not None else None
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

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
