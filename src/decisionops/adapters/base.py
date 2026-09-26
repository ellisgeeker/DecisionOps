"""Provider adapters that turn DecisionCase into DecisionPrediction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from decisionops.io import load_predictions
from decisionops.schemas import DecisionCase, DecisionPrediction


@runtime_checkable
class DecisionProvider(Protocol):
    """Uniform interface for decision model backends."""

    name: str

    def predict(self, case: DecisionCase) -> DecisionPrediction:
        """Run inference for a single case."""
        ...

    def predict_many(self, cases: list[DecisionCase]) -> list[DecisionPrediction]:
        """Run inference for many cases (default: sequential)."""
        ...


class StaticFileProvider:
    """
    Offline provider: read precomputed predictions from JSONL.

    Used for metrics development, CI, and the first technical gate
    without depending on a live model API.
    """

    def __init__(self, path: str, name: str = "static") -> None:
        preds = load_predictions(path)
        self._by_id = {p.id: p for p in preds}
        self.name = name

    def predict(self, case: DecisionCase) -> DecisionPrediction:
        try:
            pred = self._by_id[case.id]
        except KeyError as exc:
            raise KeyError(f"No static prediction for case id={case.id!r}") from exc
        # Ensure provider tag is set even if the file omitted it.
        if pred.provider is None:
            pred = pred.model_copy(update={"provider": self.name})
        return pred

    def predict_many(self, cases: list[DecisionCase]) -> list[DecisionPrediction]:
        return [self.predict(c) for c in cases]
