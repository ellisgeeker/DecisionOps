"""Dataset and prediction loaders (JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

from decisionops.schemas import DecisionCase, DecisionPrediction, OutcomeFeedback


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def load_dataset(path: str | Path) -> list[DecisionCase]:
    """Load a labeled decision dataset from JSONL."""
    raw = _read_jsonl(Path(path))
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")
    return [DecisionCase.model_validate(row) for row in raw]


def load_predictions(path: str | Path) -> list[DecisionPrediction]:
    """Load model predictions from JSONL."""
    raw = _read_jsonl(Path(path))
    if not raw:
        raise ValueError(f"Predictions file is empty: {path}")
    return [DecisionPrediction.model_validate(row) for row in raw]


def load_decisions(path: str | Path):
    """Load runtime Decision rows (e.g. from `decisionops decide -o`)."""
    # Local import: decide -> conformal -> io, so Decision cannot be top-level here.
    from decisionops.decide import Decision

    raw = _read_jsonl(Path(path))
    if not raw:
        raise ValueError(f"Decisions file is empty: {path}")
    return [Decision.model_validate(row) for row in raw]


def load_outcomes(path: str | Path) -> list[OutcomeFeedback]:
    """Load post-hoc outcome labels for recertification."""
    raw = _read_jsonl(Path(path))
    if not raw:
        raise ValueError(f"Outcomes file is empty: {path}")
    return [OutcomeFeedback.model_validate(row) for row in raw]


def align_cases_and_predictions(
    cases: list[DecisionCase],
    predictions: list[DecisionPrediction],
) -> list[tuple[DecisionCase, DecisionPrediction]]:
    """Join cases and predictions on id; require a full match."""
    case_by_id = {c.id: c for c in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("Duplicate case ids in dataset")

    pred_by_id = {p.id: p for p in predictions}
    if len(pred_by_id) != len(predictions):
        raise ValueError("Duplicate prediction ids")

    missing_preds = sorted(set(case_by_id) - set(pred_by_id))
    missing_cases = sorted(set(pred_by_id) - set(case_by_id))
    if missing_preds:
        preview = ", ".join(missing_preds[:5])
        more = f" (+{len(missing_preds) - 5} more)" if len(missing_preds) > 5 else ""
        raise ValueError(f"Missing predictions for case ids: {preview}{more}")
    if missing_cases:
        preview = ", ".join(missing_cases[:5])
        more = f" (+{len(missing_cases) - 5} more)" if len(missing_cases) > 5 else ""
        raise ValueError(f"Predictions without matching cases: {preview}{more}")

    return [(case_by_id[i], pred_by_id[i]) for i in case_by_id]
