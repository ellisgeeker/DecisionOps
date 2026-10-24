#!/usr/bin/env python3
"""
Reproducible shadow-mode loop on BANKING77 (offline).

Simulates: full-history labels (shadow) → first certificate with audit_rate →
live decide() → label-free drift → matured audit outcomes → recertify.

  uv run python scripts/run_shadow_mode_demo.py
  uv run python scripts/run_shadow_mode_demo.py --limit 800 --out /tmp/shadow-demo

Needs examples/banking77/{dataset,predictions}.jsonl (gitignored; rebuild with
scripts/prepare_hf_intent_gates.py --pack banking77).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decisionops.conformal.certify import run_certification  # noqa: E402
from decisionops.contract import DecisionContract  # noqa: E402
from decisionops.decide import decide_many, profile_from_report, save_profile  # noqa: E402
from decisionops.drift import run_drift_monitor  # noqa: E402
from decisionops.io import load_dataset, load_predictions  # noqa: E402
from decisionops.recertify import run_recertification  # noqa: E402
from decisionops.schemas import OutcomeFeedback  # noqa: E402


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(r if isinstance(r, dict) else r.model_dump(mode="json"), ensure_ascii=False)
        for r in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _contract_from_cases(cases) -> DecisionContract:
    # BANKING77 is a 77-way intent head: every class is "automatic" until a
    # product contract splits auto vs escalate. Shadow still needs a choice set
    # so recertify can validate labels.
    choices = list(cases[0].choices)
    return DecisionContract(
        name="banking77-shadow",
        description="All intents automatic; CRC on joint error. Demo contract only.",
        choices=choices,
        auto_choices=choices,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=ROOT / "examples" / "banking77",
        help="Directory with dataset.jsonl and predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "examples" / "banking77" / "shadow-demo",
        help="Output directory for profiles, decisions, reports",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap (0 = all)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--audit-rate", type=float, default=0.05)
    parser.add_argument("--audit-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drift-split",
        type=float,
        default=0.5,
        help="Baseline share of decisions",
    )
    args = parser.parse_args()

    dataset = args.pack_dir / "dataset.jsonl"
    preds_path = args.pack_dir / "predictions.jsonl"
    if not dataset.exists() or not preds_path.exists():
        print(
            f"Missing {dataset} or {preds_path}.\n"
            "Rebuild with:\n"
            "  uv run python scripts/prepare_hf_intent_gates.py --pack banking77\n"
            "  uv run python scripts/export_multi_providers.py --pack banking77",
            file=sys.stderr,
        )
        return 1

    cases = load_dataset(dataset)
    preds = load_predictions(preds_path)
    if args.limit and args.limit > 0:
        cases, preds = cases[: args.limit], preds[: args.limit]
        # Keep id alignment
        ids = {c.id for c in cases}
        preds = [p for p in preds if p.id in ids]
        case_ids = [c.id for c in cases]
        pred_by = {p.id: p for p in preds}
        preds = [pred_by[i] for i in case_ids]

    contract = _contract_from_cases(cases)
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    contract_path = out / "contract.json"
    contract_path.write_text(contract.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print(f"n={len(cases)}  alpha={args.alpha:g}  audit_rate={args.audit_rate:g}")
    print("1/5  certify (shadow labels = full history)…")
    report = run_certification(
        cases,
        preds,
        alphas=[args.alpha],
        contract=contract,
        calibrate=True,
        seed=args.seed,
    )
    level = report.levels[0]
    if not level.certificate.feasible:
        print("Certificate infeasible at this alpha; try a looser --alpha or more rows.")
        return 1
    profile = profile_from_report(
        report, args.alpha, audit_rate=args.audit_rate, audit_seed=args.audit_seed
    )
    profile_path = save_profile(profile, out / "profile.json")
    print(
        f"   profile={profile.profile_hash}  threshold={profile.threshold:.4f}  "
        f"coverage_cal={profile.coverage_calibration}"
    )

    print("2/5  decide (live routing + audit sample)…")
    decisions = decide_many(preds, profile)
    _write_jsonl(out / "decisions.jsonl", decisions)
    n_auto = sum(d.action == "auto" for d in decisions)
    n_audit = sum(d.audit for d in decisions)
    print(f"   auto={n_auto}/{len(decisions)}  audit_sample={n_audit}")

    print("3/5  drift (label-free; baseline vs held-out half of same run)…")
    # Same-run split is a smoke test only: real ops uses a time window as baseline.
    cut = max(1, int(len(decisions) * args.drift_split))
    baseline, live = decisions[:cut], decisions[cut:]
    _write_jsonl(out / "decisions_baseline.jsonl", baseline)
    _write_jsonl(out / "decisions_live.jsonl", live)
    drift = run_drift_monitor(live, baseline, profile=profile, min_window=30)
    (out / "drift_report.json").write_text(
        drift.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"   verdict={drift.verdict}  Δcertified={drift.delta_certified_rate:+.1%}  "
        f"PSI={drift.psi_confidence}"
    )

    print("4/5  outcomes for audit rows (expensive path = dataset labels)…")
    label_by_id = {c.id: c.label for c in cases}
    outcomes = [
        OutcomeFeedback(
            id=d.id,
            label=label_by_id[d.id],
            label_source="audit",
            matured_at="2026-01-01T00:00:00",
        )
        for d in decisions
        if d.audit
    ]
    # Also write a few organic escalations for diagnostics (recertify must ignore).
    organic = [
        OutcomeFeedback(
            id=d.id,
            label=label_by_id[d.id],
            label_source="escalation",
            matured_at="2026-01-01T00:00:00",
        )
        for d in decisions
        if (not d.audit) and d.action == "escalate"
    ][:50]
    _write_jsonl(out / "outcomes.jsonl", outcomes + organic)
    print(f"   audit_outcomes={len(outcomes)}  organic_extra={len(organic)} (ignored)")

    print("5/5  recertify on audit labels only…")
    recert = run_recertification(
        profile,
        decisions=decisions,
        outcomes=outcomes + organic,
        predictions=preds,
        contract=contract,
        as_of="2026-12-31",
    )
    (out / "recertify_report.json").write_text(
        recert.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if recert.profile is not None:
        save_profile(recert.profile, out / "profile_next.json")
    print(
        f"   verdict={recert.verdict}  n_audit_cal={recert.n_audit_calibration}  "
        f"feasible={recert.certificate.feasible}"
    )
    print(f"\nWrote {out}/")
    print(f"  {profile_path.name}  decisions.jsonl  drift_report.json  "
          f"outcomes.jsonl  recertify_report.json"
          + ("  profile_next.json" if recert.profile is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
