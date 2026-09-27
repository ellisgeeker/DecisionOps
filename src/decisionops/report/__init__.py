"""Report writers (console, JSON)."""

from __future__ import annotations

import json
from pathlib import Path

from decisionops.cascade import CascadeReport
from decisionops.conformal.certify import CertificationReport
from decisionops.report.console import (
    print_cascade_report,
    print_certification_report,
    print_comparison_report,
    print_eval_report,
    print_gate_report,
)
from decisionops.report.html import write_html_report
from decisionops.schemas import ComparisonReport, EvalResult, GateReport


def write_json_report(
    result: EvalResult | GateReport | ComparisonReport | CertificationReport | CascadeReport,
    path: str | Path,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


__all__ = [
    "print_cascade_report",
    "print_certification_report",
    "print_comparison_report",
    "print_eval_report",
    "print_gate_report",
    "write_html_report",
    "write_json_report",
]
