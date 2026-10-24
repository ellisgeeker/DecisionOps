"""Rich console summaries."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from decisionops.cascade import CascadeReport
from decisionops.conformal.certify import CertificationReport
from decisionops.schemas import ComparisonReport, EvalResult, GateReport


def print_eval_report(result: EvalResult, console: Console | None = None) -> None:
    console = console or Console()

    console.print()
    console.rule("[bold]DecisionOps Eval[/bold]")
    console.print(
        f"samples={result.n_samples}"
        + (f"  provider={result.provider}" if result.provider else "")
        + (f"  model={result.model}" if result.model else "")
    )
    protocol = (result.metadata or {}).get("protocol")
    if protocol:
        console.print(f"[dim]protocol={protocol}[/dim]")
    console.print()

    metrics = Table(title="Core Metrics", show_header=True, header_style="bold")
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")
    metrics.add_row("Accuracy", f"{result.accuracy:.4f}")
    metrics.add_row("ECE", f"{result.ece:.4f}")
    metrics.add_row("Brier", f"{result.brier:.4f}")
    metrics.add_row("NLL", f"{result.nll:.4f}")
    console.print(metrics)

    if result.latency.count > 0:
        lat = Table(title="Latency (ms)", show_header=True, header_style="bold")
        lat.add_column("Stat")
        lat.add_column("Value", justify="right")
        lat.add_row("count", str(result.latency.count))
        if result.latency.p50_ms is not None:
            lat.add_row("P50", f"{result.latency.p50_ms:.1f}")
        if result.latency.p95_ms is not None:
            lat.add_row("P95", f"{result.latency.p95_ms:.1f}")
        if result.latency.p99_ms is not None:
            lat.add_row("P99", f"{result.latency.p99_ms:.1f}")
        if result.latency.mean_ms is not None:
            lat.add_row("mean", f"{result.latency.mean_ms:.1f}")
        console.print(lat)

    if result.thresholds:
        thr = Table(
            title="Empirical Thresholds (not conformal-certified)",
            show_header=True,
            header_style="bold",
        )
        thr.add_column("Target Risk")
        thr.add_column("Threshold", justify="right")
        thr.add_column("Coverage", justify="right")
        thr.add_column("Sel. Acc", justify="right")
        thr.add_column("Sel. Risk", justify="right")
        thr.add_column("Accepted", justify="right")
        for t in result.thresholds:
            thr.add_row(
                f"{t.target_risk:.2%}",
                f"{t.threshold:.3f}",
                f"{t.coverage:.1%}",
                f"{t.selective_accuracy:.4f}",
                f"{t.selective_risk:.4f}",
                f"{t.n_accepted}/{t.n_total}",
            )
        console.print(thr)
        console.print(
            "[dim]Same-set path: thresholds fit and scored on one labeled pool. "
            "Prefer `decisionops gate` (cal/test split) for serious decisions.[/dim]"
        )

    console.print()


def print_gate_report(report: GateReport, console: Console | None = None) -> None:
    console = console or Console()

    console.print()
    console.rule("[bold]DecisionOps Technical Gate[/bold]")
    console.print(
        f"n_total={report.split.n_total}  "
        f"n_cal={report.split.n_calibration}  "
        f"n_test={report.split.n_test}  "
        f"cal_ratio={report.split.cal_ratio:.2f}  "
        f"seed={report.split.seed}"
    )
    if report.provider or report.model:
        console.print(
            (f"provider={report.provider}  " if report.provider else "")
            + (f"model={report.model}" if report.model else "")
        )
    if report.temperature is not None:
        console.print(
            f"temperature T={report.temperature:.4f}  "
            f"(fit NLL {report.temperature_fit.get('nll_before', float('nan')):.4f} "
            f"→ {report.temperature_fit.get('nll_after', float('nan')):.4f})"
        )
    if report.isotonic_fit:
        console.print(
            "isotonic  "
            f"(fit NLL {report.isotonic_fit.get('nll_before', float('nan')):.4f} "
            f"→ {report.isotonic_fit.get('nll_after', float('nan')):.4f})"
        )
    console.print()

    metrics = Table(
        title="Test-set metrics (thresholds chosen on calibration only)",
        show_header=True,
        header_style="bold",
    )
    metrics.add_column("Condition")
    metrics.add_column("Accuracy", justify="right")
    metrics.add_column("ECE", justify="right")
    metrics.add_column("Brier", justify="right")
    metrics.add_column("NLL", justify="right")
    for c in report.conditions:
        metrics.add_row(
            c.name,
            f"{c.accuracy:.4f}",
            f"{c.ece:.4f}",
            f"{c.brier:.4f}",
            f"{c.nll:.4f}",
        )
    console.print(metrics)

    thr = Table(
        title="Test coverage at calibration-chosen thresholds",
        show_header=True,
        header_style="bold",
    )
    thr.add_column("Condition")
    thr.add_column("Target Risk")
    thr.add_column("Thr (cal)", justify="right")
    thr.add_column("Coverage (test)", justify="right")
    thr.add_column("Sel. Risk (test)", justify="right")
    thr.add_column("Accepted", justify="right")
    for c in report.conditions:
        for t in c.thresholds:
            thr.add_row(
                c.name,
                f"{t.target_risk:.2%}",
                f"{t.threshold:.3f}",
                f"{t.coverage:.1%}",
                f"{t.selective_risk:.4f}",
                f"{t.n_accepted}/{t.n_total}",
            )
    console.print(thr)

    v = report.verdict
    color = {"go": "green", "watch": "yellow", "no-go": "red"}.get(v.recommendation, "white")
    console.print()
    console.print(f"[bold {color}]Verdict: {v.recommendation.upper()}[/bold {color}]")
    console.print(
        f"Basis: CRC certificate at {v.verdict_alpha:.0%} joint automated error rate; "
        f"go needs certified test coverage >= {v.min_coverage:.0%}"
    )
    if v.certified_feasible and v.certified_coverage is not None:
        console.print(
            f"Certified @ {v.verdict_alpha:.0%}: threshold={v.certified_threshold:.4f}  "
            f"coverage (test)={v.certified_coverage:.1%}  "
            f"joint err (test)={v.certified_joint_error:.4f}"
        )
    else:
        console.print(f"Certified @ {v.verdict_alpha:.0%}: [red]infeasible[/red]")
    for note in v.notes:
        console.print(f"• {escape(note)}")
    console.print(
        "[dim]ECE and the empirical thresholds above are context, not the verdict. "
        "The coverage bar is a policy choice (--min-coverage); tune it to your cost "
        "of escalation.[/dim]"
    )
    console.print()


def print_certification_report(
    report: CertificationReport, console: Console | None = None
) -> None:
    console = console or Console()

    console.print()
    console.rule("[bold]DecisionOps Certification (Conformal Risk Control)[/bold]")
    console.print(
        f"n_total={report.split.n_total}  "
        f"n_crc_cal={report.split.n_calibration}  "
        f"n_test={report.split.n_test}  "
        f"cal_ratio={report.split.cal_ratio:.2f}  "
        f"seed={report.split.seed}"
    )
    if report.provider or report.model:
        console.print(
            (f"provider={report.provider}  " if report.provider else "")
            + (f"model={report.model}" if report.model else "")
        )
    if report.temperature is not None:
        console.print(
            f"temperature T={report.temperature:.4f}  "
            f"(fit on {report.n_temperature_fit} samples, disjoint from the CRC set)"
        )
    else:
        console.print("[dim]temperature: not applied[/dim]")

    level_meta = (report.metadata or {}).get("confidence_level")
    high_prob = level_meta is not None
    if high_prob:
        console.print(
            f"guarantee: [bold]high-probability[/bold] — P(joint error <= alpha) "
            f">= {float(level_meta):.0%} over the calibration draw (Clopper-Pearson)"
        )
    else:
        console.print(r"guarantee: in expectation — E\[joint error] <= alpha (CRC)")
    loss_bound = (report.metadata or {}).get("loss_bound")
    contract_meta = (report.metadata or {}).get("contract") or {}
    severity = contract_meta.get("severity") or {}
    if loss_bound is not None and (float(loss_bound) != 1.0 or severity):
        sev_txt = (
            ", ".join(f"{k}={v:g}" for k, v in sorted(severity.items()))
            if severity
            else "uniform 1"
        )
        console.print(
            f"loss bound B={float(loss_bound):g}  "
            f"(miss severity by true label: {sev_txt})"
        )
    console.print()

    cert = Table(
        title="Certified thresholds (joint automated error <= alpha)",
        show_header=True,
        header_style="bold",
    )
    cert.add_column("Target alpha")
    cert.add_column("Certified", justify="center")
    cert.add_column("Threshold", justify="right")
    cert.add_column("Upper bound" if high_prob else "CRC bound", justify="right")
    cert.add_column("Coverage (cal)", justify="right")
    cert.add_column("n_cal / min_n", justify="right")

    for level in report.levels:
        c = level.certificate
        if not c.feasible:
            cert.add_row(
                f"{c.alpha:.2%}",
                "[red]INFEASIBLE[/red]",
                "—",
                "—",
                "—",
                f"{c.n_calibration} / {c.min_n_required}",
            )
            continue
        cert.add_row(
            f"{c.alpha:.2%}",
            "[green]yes[/green]",
            f"{c.threshold:.4f}",
            "—" if c.crc_bound_value is None else f"{c.crc_bound_value:.4f}",
            "—" if c.coverage_calibration is None else f"{c.coverage_calibration:.1%}",
            f"{c.n_calibration} / {c.min_n_required}",
        )
    console.print(cert)

    feasible = [lv for lv in report.levels if lv.certificate.feasible and lv.test]
    if feasible:
        held = Table(
            title="Held-out measurement at the certified threshold",
            show_header=True,
            header_style="bold",
        )
        held.add_column("Target alpha")
        held.add_column("Coverage (test)", justify="right")
        held.add_column("Joint err (test)", justify="right")
        held.add_column("Sel. risk (test)", justify="right")
        held.add_column("Accepted", justify="right")
        held.add_column("<= alpha?", justify="center")
        for level in feasible:
            t = level.test or {}
            ok = level.bound_held_on_test
            held.add_row(
                f"{level.certificate.alpha:.2%}",
                f"{float(t['coverage']):.1%}",
                f"{float(t['joint_error_rate']):.4f}",
                f"{float(t['selective_risk']):.4f}",
                f"{int(t['n_accepted'])}/{int(t['n_total'])}",
                "[green]yes[/green]" if ok else "[yellow]no[/yellow]",
            )
        console.print(held)
        if high_prob:
            console.print(
                "[dim]The bound is on true risk, not on this finite test split; "
                "a split can still land above alpha by sampling noise.[/dim]"
            )
        else:
            console.print(
                "[dim]A single test split landing above alpha is expected roughly half "
                r"the time — the guarantee is on E\[loss]. Judge by averaging over seeds.[/dim]"
            )

    infeasible = [lv for lv in report.levels if not lv.certificate.feasible]
    if infeasible:
        console.print()
        console.print(
            f"[bold red]No certificate at "
            f"{', '.join(f'{lv.certificate.alpha:.0%}' for lv in infeasible)}.[/bold red] "
            "Abstain on all traffic at those levels."
        )
        for level in infeasible:
            for note in level.certificate.notes:
                console.print(f"  [red]•[/red] {escape(note)}")

    console.print()
    for note in report.notes:
        console.print(f"• {escape(note)}")
    console.print()


def _risk_key(risk: float) -> str:
    return f"{risk:.4f}".rstrip("0").rstrip(".")


def print_comparison_report(
    report: ComparisonReport, console: Console | None = None
) -> None:
    console = console or Console()

    console.print()
    console.rule("[bold]DecisionOps Model Comparison[/bold]")
    n_crc = (report.metadata or {}).get("n_crc_calibration")
    crc_bit = f"  n_crc_cal={n_crc}" if n_crc is not None else ""
    console.print(
        f"dataset={report.dataset}  "
        f"n_cal={report.split.n_calibration}  "
        f"n_test={report.split.n_test}  "
        f"seed={report.split.seed}{crc_bit}  "
        f"(shared split across models)"
    )
    console.print(
        "[dim]Certified quantity is the joint automated error rate, "
        "not conditional selective risk.[/dim]"
    )
    console.print()

    quality = Table(
        title="Test quality (calibrated condition where available)",
        show_header=True,
        header_style="bold",
    )
    quality.add_column("Model")
    quality.add_column("Acc", justify="right")
    quality.add_column("ECE raw", justify="right")
    quality.add_column("ECE cal", justify="right")
    quality.add_column("Brier cal", justify="right")
    quality.add_column("NLL cal", justify="right")
    quality.add_column("T", justify="right")
    for e in report.entries:
        quality.add_row(
            e.name,
            f"{e.accuracy:.4f}",
            f"{e.ece_raw:.4f}",
            "—" if e.ece_calibrated is None else f"{e.ece_calibrated:.4f}",
            "—" if e.brier_calibrated is None else f"{e.brier_calibrated:.4f}",
            "—" if e.nll_calibrated is None else f"{e.nll_calibrated:.4f}",
            "—" if e.temperature is None else f"{e.temperature:.3f}",
        )
    console.print(quality)

    cert = Table(
        title="Certified coverage (CRC; joint automated error rate)",
        show_header=True,
        header_style="bold",
    )
    cert.add_column("Model", no_wrap=True)
    for risk in report.target_risks:
        cert.add_column(f"@{risk:.0%} cov (jerr)", justify="right", no_wrap=True)
    cert.add_column("P95 ms", justify="right")
    cert.add_column("Verdict", no_wrap=True)

    for e in report.entries:
        row = [e.name]
        for risk in report.target_risks:
            key = _risk_key(risk)
            if not e.certified_feasible_by_risk.get(key):
                row.append("[red]infeasible[/red]")
                continue
            c = e.certified_coverage_by_risk.get(key)
            j = e.certified_joint_error_by_risk.get(key)
            if c is None:
                row.append("—")
            elif j is None:
                row.append(f"{c:.1%}")
            else:
                row.append(f"{c:.1%} ({j:.3f})")
        row.append("—" if e.latency_p95_ms is None else f"{e.latency_p95_ms:.2f}")
        row.append(e.verdict)
        cert.add_row(*row)
    console.print(cert)

    emp = Table(
        title="Empirical coverage (not certified; ! = test selective risk > target)",
        show_header=True,
        header_style="bold",
    )
    emp.add_column("Model", no_wrap=True)
    for risk in report.target_risks:
        emp.add_column(f"@{risk:.0%} cov (risk)", justify="right", no_wrap=True)

    for e in report.entries:
        row = [e.name]
        for risk in report.target_risks:
            key = _risk_key(risk)
            c = e.coverage_by_risk.get(key)
            r = e.selective_risk_by_risk.get(key)
            if c is None:
                row.append("—")
                continue
            if r is None:
                row.append(f"{c:.1%}")
            else:
                marker = "!" if r > risk + 1e-9 else ""
                row.append(f"{c:.1%} ({r:.3f}){marker}")
        emp.add_row(*row)
    console.print(emp)

    if report.best_by_coverage_at:
        console.print()
        counts = (report.metadata or {}).get("n_feasible_by_risk", {})
        for risk in report.target_risks:
            key = _risk_key(risk)
            winner = report.best_by_coverage_at.get(key)
            if not winner:
                console.print(
                    f"@{risk:.0%} risk: [red]no feasible certificate[/red] — abstain"
                )
                continue
            entry = next((e for e in report.entries if e.name == winner), None)
            cov_v = entry.certified_coverage_by_risk.get(key) if entry else None
            n_ok = counts.get(key)
            suffix = ""
            if isinstance(n_ok, int):
                suffix = (
                    f"  [dim]({n_ok}/{len(report.entries)} "
                    f"had a feasible certificate)[/dim]"
                )
            cov_txt = "" if cov_v is None else f" at {cov_v:.1%} certified coverage"
            console.print(
                f"Most automation with a @{risk:.0%} CRC certificate: "
                f"[bold]{winner}[/bold]{cov_txt}{suffix}"
            )
    for note in report.notes:
        console.print(f"• {escape(note)}")
    console.print()


def print_cascade_report(report: CascadeReport, console: Console | None = None) -> None:
    console = console or Console()

    console.print()
    console.rule("[bold]DecisionOps Cascade (cost vs miss rate)[/bold]")
    console.print(
        f"n_total={report.split.n_total}  "
        f"n_cal={report.split.n_calibration}  "
        f"n_test={report.split.n_test}  "
        f"cal_ratio={report.split.cal_ratio:.2f}  "
        f"seed={report.split.seed}"
    )
    if report.provider or report.model:
        console.print(
            (f"provider={report.provider}  " if report.provider else "")
            + (f"model={report.model}" if report.model else "")
        )
    cname = report.contract.get("name", "?")
    auto = report.contract.get("auto_choices", [])
    esc = report.contract.get("escalate_choices", [])
    console.print(
        f"contract={cname}  auto={auto}  escalate={esc}"
    )
    console.print(
        f"baseline mean cost (always escalate)={report.baseline_mean_cost:.4f}  "
        f"always-auto mean={report.always_auto_mean_cost:.4f}  "
        f"always-auto miss={report.always_auto_miss_rate:.4f}"
    )
    console.print()

    table = Table(
        title="Cost–miss curve (held-out test; oracle escalate pricing)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Threshold", justify="right")
    table.add_column("Automation", justify="right")
    table.add_column("Miss rate", justify="right")
    table.add_column("Sel. risk", justify="right")
    table.add_column("Mean cost", justify="right")
    table.add_column("Cost saved", justify="right")
    table.add_column("Saved %", justify="right")
    table.add_column("Auto / N", justify="right")

    # Print a sparse grid: endpoints + every ~5th point, preferring round thresholds.
    if not report.curve:
        console.print("[yellow]No curve points.[/yellow]")
    else:
        n = len(report.curve)
        keep = {0, n - 1}
        for i, p in enumerate(report.curve):
            hundredths = int(round(p.threshold * 100))
            on_grid = abs(p.threshold * 100 - hundredths) < 1e-9
            if on_grid and hundredths % 5 == 0:
                keep.add(i)
            if i % max(1, n // 12) == 0:
                keep.add(i)
        for i in sorted(keep):
            p = report.curve[i]
            sel = "—" if p.selective_risk is None else f"{p.selective_risk:.4f}"
            frac = "—" if p.cost_saved_frac is None else f"{p.cost_saved_frac:.1%}"
            table.add_row(
                f"{p.threshold:.2f}",
                f"{p.automation_rate:.1%}",
                f"{p.miss_rate:.4f}",
                sel,
                f"{p.mean_cost:.4f}",
                f"{p.cost_saved:.4f}",
                frac,
                f"{p.n_auto}/{p.n_total}",
            )
        console.print(table)

    console.print(
        "[dim]Escalation costs are oracle. miss_rate is the joint automated error rate "
        "(auto AND wrong over all traffic) — the same quantity CRC certifies. "
        "Thresholds here are empirical, not certified.[/dim]"
    )
    for note in report.notes:
        console.print(f"• {escape(str(note))}")
    console.print()
