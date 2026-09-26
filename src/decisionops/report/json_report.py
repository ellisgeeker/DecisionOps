"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path

from decisionops.schemas import EvalResult, GateReport


def write_json_report(result: EvalResult | GateReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
