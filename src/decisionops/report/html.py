"""Self-contained HTML reports. No CDN, no JS required to read the two core charts."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from decisionops.cascade import CascadeReport
from decisionops.conformal.certify import CertificationReport
from decisionops.schemas import ComparisonReport, EvalResult, GateReport

Report = EvalResult | GateReport | ComparisonReport | CertificationReport | CascadeReport

_W = 440
_H = 320
_PAD_L = 48
_PAD_R = 16
_PAD_T = 16
_PAD_B = 40


def write_html_report(result: Report, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(result), encoding="utf-8")
    return out


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.1%}"


def _xy(x: float, y: float) -> tuple[float, float]:
    px = _PAD_L + x * (_W - _PAD_L - _PAD_R)
    py = _PAD_T + (1.0 - y) * (_H - _PAD_T - _PAD_B)
    return px, py


def _axes(x_label: str, y_label: str) -> str:
    x0, y0 = _xy(0, 0)
    x1, y1 = _xy(1, 1)
    ticks = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        tx, _ = _xy(t, 0)
        _, ty = _xy(0, t)
        ticks.append(
            f'<line x1="{tx:.1f}" y1="{y0:.1f}" x2="{tx:.1f}" y2="{y0 + 4:.1f}" />'
            f'<text x="{tx:.1f}" y="{y0 + 16:.1f}" text-anchor="middle">{t:.2f}</text>'
            f'<line x1="{x0:.1f}" y1="{ty:.1f}" x2="{x0 - 4:.1f}" y2="{ty:.1f}" />'
            f'<text x="{x0 - 8:.1f}" y="{ty + 4:.1f}" text-anchor="end">{t:.2f}</text>'
        )
    return f"""
    <rect x="0" y="0" width="{_W}" height="{_H}" class="plot-bg" />
    <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" />
    <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" />
    {"".join(ticks)}
    <text x="{(_PAD_L + _W - _PAD_R) / 2:.1f}" y="{_H - 6}" text-anchor="middle">{_esc(x_label)}</text>
    <text transform="translate(14 {(_PAD_T + _H - _PAD_B) / 2:.1f}) rotate(-90)" text-anchor="middle">{_esc(y_label)}</text>
    """


def reliability_svg(bins: list[dict[str, Any]], *, title: str) -> str:
    """Predicted confidence vs actual accuracy. Perfect calibration is the diagonal."""
    x0, y0 = _xy(0, 0)
    x1, y1 = _xy(1, 1)
    marks = [
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" class="diag" />'
    ]
    populated = [b for b in bins if int(b.get("count") or 0) > 0]
    max_n = max((int(b["count"]) for b in populated), default=1)
    for b in populated:
        conf = float(b["avg_confidence"])
        acc = float(b["accuracy"])
        if not (math.isfinite(conf) and math.isfinite(acc)):
            continue
        px, py = _xy(conf, acc)
        r = 4 + 8 * math.sqrt(int(b["count"]) / max_n)
        lo, hi = float(b["bin_low"]), float(b["bin_high"])
        n = int(b["count"])
        marks.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" class="dot">'
            f"<title>[{lo:.2f},{hi:.2f}] n={n} conf={conf:.3f} acc={acc:.3f}</title>"
            f"</circle>"
        )
    return f"""
    <figure>
      <figcaption>{_esc(title)}</figcaption>
      <svg viewBox="0 0 {_W} {_H}" role="img" aria-label="{_esc(title)}">
        {_axes("predicted confidence", "actual accuracy")}
        {"".join(marks)}
      </svg>
      <p class="hint">Dot size ∝ bin count. On the diagonal, confidence matches accuracy.</p>
    </figure>
    """


def risk_coverage_svg(
    points: list[dict[str, Any]],
    *,
    title: str,
    target_risks: list[float] | None = None,
) -> str:
    usable = [
        p
        for p in points
        if int(p.get("n_accepted") or 0) > 0
        and math.isfinite(float(p.get("coverage", float("nan"))))
        and math.isfinite(float(p.get("selective_risk", float("nan"))))
    ]
    usable.sort(key=lambda p: float(p["coverage"]))
    lines = []
    for risk in target_risks or []:
        _, y = _xy(0, risk)
        x1, _ = _xy(1, risk)
        x0, _ = _xy(0, risk)
        lines.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" class="target" />'
            f'<text x="{x1 - 4:.1f}" y="{y - 4:.1f}" text-anchor="end" class="target-label">'
            f"{risk:.0%} risk</text>"
        )
    path = []
    dots = []
    for p in usable:
        px, py = _xy(float(p["coverage"]), float(p["selective_risk"]))
        path.append(f"{px:.1f},{py:.1f}")
        dots.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" class="dot-sm">'
            f"<title>t={float(p['threshold']):.2f} cov={float(p['coverage']):.1%} "
            f"risk={float(p['selective_risk']):.3f}</title></circle>"
        )
    poly = ""
    if len(path) >= 2:
        poly = f'<polyline points="{" ".join(path)}" class="curve" />'
    return f"""
    <figure>
      <figcaption>{_esc(title)}</figcaption>
      <svg viewBox="0 0 {_W} {_H}" role="img" aria-label="{_esc(title)}">
        {_axes("coverage (auto-accepted)", "selective risk")}
        {"".join(lines)}
        {poly}
        {"".join(dots)}
      </svg>
      <p class="hint">More automation is to the right. Higher error is up. Empirical, not CRC.</p>
    </figure>
    """


def cost_miss_svg(
    points: list[dict[str, Any]],
    *,
    title: str,
    baseline_mean_cost: float | None = None,
) -> str:
    """Scatter/line of cost_saved (x) vs miss_rate (y)."""
    usable = [
        p
        for p in points
        if math.isfinite(float(p.get("cost_saved", float("nan"))))
        and math.isfinite(float(p.get("miss_rate", float("nan"))))
    ]
    usable.sort(key=lambda p: float(p["cost_saved"]))
    if not usable:
        return f"<figure><figcaption>{_esc(title)}</figcaption><p class=\"hint\">No points.</p></figure>"

    xs = [float(p["cost_saved"]) for p in usable]
    ys = [float(p["miss_rate"]) for p in usable]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys) if max(ys) > 0 else 0.01
    if abs(x_max - x_min) < 1e-12:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 0.01

    def xy(x: float, y: float) -> tuple[float, float]:
        px = _PAD_L + (x - x_min) / (x_max - x_min) * (_W - _PAD_L - _PAD_R)
        py = _PAD_T + (1.0 - (y - y_min) / (y_max - y_min)) * (_H - _PAD_T - _PAD_B)
        return px, py

    path = []
    dots = []
    for p in usable:
        px, py = xy(float(p["cost_saved"]), float(p["miss_rate"]))
        path.append(f"{px:.1f},{py:.1f}")
        dots.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" class="dot-sm">'
            f"<title>t={float(p['threshold']):.2f} auto={float(p['automation_rate']):.1%} "
            f"miss={float(p['miss_rate']):.4f} saved={float(p['cost_saved']):.3f}</title></circle>"
        )
    poly = ""
    if len(path) >= 2:
        poly = f'<polyline points="{" ".join(path)}" class="curve" />'

    # axes labels (manual, since _axes assumes 0..1)
    axes = (
        f'<rect x="{_PAD_L}" y="{_PAD_T}" width="{_W - _PAD_L - _PAD_R}" '
        f'height="{_H - _PAD_T - _PAD_B}" class="plot-bg" />'
        f'<line x1="{_PAD_L}" y1="{_H - _PAD_B}" x2="{_W - _PAD_R}" y2="{_H - _PAD_B}" />'
        f'<line x1="{_PAD_L}" y1="{_PAD_T}" x2="{_PAD_L}" y2="{_H - _PAD_B}" />'
        f'<text x="{(_PAD_L + _W - _PAD_R) / 2:.1f}" y="{_H - 8}" text-anchor="middle">'
        f"cost saved vs always-escalate</text>"
        f'<text x="12" y="{(_PAD_T + _H - _PAD_B) / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 12,{(_PAD_T + _H - _PAD_B) / 2:.1f})">miss rate (joint)</text>'
        f'<text x="{_PAD_L}" y="{_H - _PAD_B + 14}" text-anchor="start">{x_min:.2f}</text>'
        f'<text x="{_W - _PAD_R}" y="{_H - _PAD_B + 14}" text-anchor="end">{x_max:.2f}</text>'
        f'<text x="{_PAD_L - 4}" y="{_H - _PAD_B}" text-anchor="end">{y_min:.2f}</text>'
        f'<text x="{_PAD_L - 4}" y="{_PAD_T + 8}" text-anchor="end">{y_max:.2f}</text>'
    )
    hint = "More savings to the right. Higher automated miss rate is up. Empirical, oracle escalate."
    if baseline_mean_cost is not None:
        hint += f" Baseline escalate mean={baseline_mean_cost:.3f}."
    return f"""
    <figure>
      <figcaption>{_esc(title)}</figcaption>
      <svg viewBox="0 0 {_W} {_H}" role="img" aria-label="{_esc(title)}">
        {axes}
        {poly}
        {"".join(dots)}
      </svg>
      <p class="hint">{_esc(hint)}</p>
    </figure>
    """



def _css() -> str:
    return """
    :root { color-scheme: light dark; --fg: #1a1a1a; --muted: #5c5c5c; --bg: #f7f7f4; --card: #fff; --line: #222; --accent: #0b57d0; --warn: #a15c00; }
    @media (prefers-color-scheme: dark) {
      :root { --fg: #eee; --muted: #aaa; --bg: #161616; --card: #1e1e1e; --line: #ddd; --accent: #8ab4f8; --warn: #f5c16c; }
    }
    body { margin: 0 auto; max-width: 960px; padding: 24px 20px 64px; font: 15px/1.45 system-ui, sans-serif; color: var(--fg); background: var(--bg); }
    h1 { font-size: 1.5rem; margin: 0 0 4px; }
    h2 { font-size: 1.1rem; margin: 28px 0 8px; }
    .sub, .hint, figcaption { color: var(--muted); }
    .hint { font-size: 0.85rem; margin: 4px 0 0; }
    table { border-collapse: collapse; width: 100%; background: var(--card); }
    th, td { border-bottom: 1px solid color-mix(in srgb, var(--fg) 12%, transparent); padding: 6px 10px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
    figure { margin: 0; background: var(--card); padding: 8px 8px 12px; }
    svg { width: 100%; height: auto; }
    svg line, svg polyline { stroke: var(--line); fill: none; stroke-width: 1; }
    svg .diag { stroke-dasharray: 4 4; opacity: 0.6; }
    svg .curve { stroke: var(--accent); stroke-width: 2; }
    svg .target { stroke: var(--warn); stroke-dasharray: 3 3; }
    svg .target-label { fill: var(--warn); font-size: 10px; }
    svg .dot { fill: var(--accent); fill-opacity: 0.75; stroke: none; }
    svg .dot-sm { fill: var(--accent); }
    svg text { fill: var(--muted); font-size: 10px; }
    svg .plot-bg { fill: transparent; stroke: none; }
    .verdict { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; }
    .verdict.go { background: #d1f0d5; color: #0d5c1f; }
    .verdict.watch { background: #ffe7c2; color: #7a4b00; }
    .verdict.no-go { background: #ffd5d5; color: #8b1212; }
    ul.notes { padding-left: 18px; }
    """


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        f"<meta charset=\"utf-8\" />\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"<title>{_esc(title)}</title>\n<style>{_css()}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _gate_html(report: GateReport) -> str:
    focus = next((c for c in report.conditions if c.name == "temperature_scaled"), None)
    iso = next((c for c in report.conditions if c.name == "isotonic"), None)
    raw = next((c for c in report.conditions if c.name == "raw"), report.conditions[0])
    charts = []
    if raw.reliability_bins:
        charts.append(reliability_svg(raw.reliability_bins, title="Reliability (raw)"))
    if focus and focus.reliability_bins:
        charts.append(
            reliability_svg(focus.reliability_bins, title="Reliability (temperature-scaled)")
        )
    if iso and iso.reliability_bins:
        charts.append(reliability_svg(iso.reliability_bins, title="Reliability (isotonic)"))
    curve_src = focus or iso or raw
    if curve_src.risk_coverage:
        charts.append(
            risk_coverage_svg(
                curve_src.risk_coverage,
                title="Risk–coverage (test, empirical)",
                target_risks=report.target_risks,
            )
        )

    rows = []
    for c in report.conditions:
        rows.append(
            "<tr>"
            f"<td>{_esc(c.name)}</td>"
            f"<td>{_fmt(c.accuracy)}</td>"
            f"<td>{_fmt(c.ece)}</td>"
            f"<td>{_fmt(c.brier)}</td>"
            f"<td>{_fmt(c.nll)}</td>"
            "</tr>"
        )
    thr_rows = []
    for c in report.conditions:
        for t in c.thresholds:
            thr_rows.append(
                "<tr>"
                f"<td>{_esc(c.name)}</td>"
                f"<td>{t.target_risk:.0%}</td>"
                f"<td>{_fmt(t.threshold, 3)}</td>"
                f"<td>{_pct(t.coverage)}</td>"
                f"<td>{_fmt(t.selective_risk)}</td>"
                f"<td>{t.n_accepted}/{t.n_total}</td>"
                "</tr>"
            )

    who = " ".join(
        p
        for p in (
            f"provider={report.provider}" if report.provider else "",
            f"model={report.model}" if report.model else "",
        )
        if p
    )
    t_line = (
        f"temperature T={report.temperature:.4f}"
        if report.temperature is not None
        else "temperature not applied"
    )
    rec = report.verdict.recommendation
    notes = "".join(f"<li>{_esc(n)}</li>" for n in report.verdict.notes)

    return _page(
        "DecisionOps Gate",
        f"""
        <h1>DecisionOps technical gate</h1>
        <p class="sub">n_total={report.split.n_total}
          n_cal={report.split.n_calibration}
          n_test={report.split.n_test}
          seed={report.split.seed}
          { _esc(who) }</p>
        <p class="sub">{_esc(t_line)}. Thresholds below are <strong>empirical</strong>
          (fit on calibration). For a finite-sample bound use
          <code>decisionops certify</code>.</p>
        <p>Verdict: <span class="verdict { _esc(rec) }">{_esc(rec)}</span>
          <span class="hint">heuristic, not a certificate</span></p>
        <h2>Test metrics</h2>
        <table>
          <thead><tr><th>Condition</th><th>Accuracy</th><th>ECE</th><th>Brier</th><th>NLL</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        <h2>Charts</h2>
        <div class="charts">{"".join(charts)}</div>
        <h2>Empirical thresholds on test</h2>
        <table>
          <thead><tr><th>Condition</th><th>Target</th><th>Thr (cal)</th><th>Coverage</th><th>Sel. risk</th><th>Accepted</th></tr></thead>
          <tbody>{"".join(thr_rows)}</tbody>
        </table>
        <h2>Notes</h2>
        <ul class="notes">{notes}</ul>
        """,
    )


def _eval_html(result: EvalResult) -> str:
    bins = (result.metadata or {}).get("reliability_bins") or []
    curve = (result.metadata or {}).get("risk_coverage") or []
    charts = []
    if bins:
        charts.append(reliability_svg(bins, title="Reliability (same-set)"))
    if curve:
        charts.append(
            risk_coverage_svg(curve, title="Risk–coverage (same-set, empirical)")
        )
    thr = "".join(
        "<tr>"
        f"<td>{t.target_risk:.0%}</td>"
        f"<td>{_fmt(t.threshold, 3)}</td>"
        f"<td>{_pct(t.coverage)}</td>"
        f"<td>{_fmt(t.selective_risk)}</td>"
        f"<td>{t.n_accepted}/{t.n_total}</td>"
        "</tr>"
        for t in result.thresholds
    )
    return _page(
        "DecisionOps Eval",
        f"""
        <h1>DecisionOps eval</h1>
        <p class="sub">n={result.n_samples} protocol=same-set.
          Thresholds fit and scored on one pool — prefer <code>gate</code>.</p>
        <table>
          <tbody>
            <tr><td>Accuracy</td><td>{_fmt(result.accuracy)}</td></tr>
            <tr><td>ECE</td><td>{_fmt(result.ece)}</td></tr>
            <tr><td>Brier</td><td>{_fmt(result.brier)}</td></tr>
            <tr><td>NLL</td><td>{_fmt(result.nll)}</td></tr>
          </tbody>
        </table>
        <div class="charts">{"".join(charts)}</div>
        <h2>Empirical thresholds</h2>
        <table>
          <thead><tr><th>Target</th><th>Threshold</th><th>Coverage</th><th>Sel. risk</th><th>Accepted</th></tr></thead>
          <tbody>{thr}</tbody>
        </table>
        """,
    )


def _cert_html(report: CertificationReport) -> str:
    level_meta = (report.metadata or {}).get("confidence_level")
    if level_meta is None:
        guarantee = "Guarantee: E[joint error] ≤ α in expectation (CRC)."
        split_hint = "A single split landing above α is expected roughly half the time."
    else:
        guarantee = (
            f"Guarantee: P(joint error ≤ α) ≥ {float(level_meta):.0%} over the "
            "calibration draw (Clopper-Pearson)."
        )
        split_hint = "The bound is on true risk; a finite test split can still land above α."
    rows = []
    for level in report.levels:
        c = level.certificate
        if not c.feasible or level.test is None:
            rows.append(
                "<tr>"
                f"<td>{c.alpha:.0%}</td><td>INFEASIBLE</td><td>—</td><td>—</td><td>—</td><td>—</td>"
                "</tr>"
            )
            continue
        t = level.test
        rows.append(
            "<tr>"
            f"<td>{c.alpha:.0%}</td>"
            f"<td>{_fmt(c.threshold, 4)}</td>"
            f"<td>{_pct(float(t['coverage']))}</td>"
            f"<td>{_fmt(float(t['joint_error_rate']))}</td>"
            f"<td>{_fmt(float(t['selective_risk']))}</td>"
            f"<td>{'yes' if level.bound_held_on_test else 'no'}</td>"
            "</tr>"
        )
    notes = "".join(f"<li>{_esc(n)}</li>" for n in report.notes)
    return _page(
        "DecisionOps Certification",
        f"""
        <h1>DecisionOps certification (CRC)</h1>
        <p class="sub">n_crc_cal={report.split.n_calibration} n_test={report.split.n_test}
          seed={report.split.seed}. Certified quantity is the
          <strong>joint automated error rate</strong>, not selective risk.
          {_esc(guarantee)}</p>
        <table>
          <thead><tr><th>α</th><th>λ̂</th><th>Coverage (test)</th><th>Joint err</th><th>Sel. risk</th><th>≤ α this split?</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        <p class="hint">{_esc(split_hint)}</p>
        <ul class="notes">{notes}</ul>
        """,
    )


def _compare_html(report: ComparisonReport) -> str:
    def rk(risk: float) -> str:
        return f"{risk:.4f}".rstrip("0").rstrip(".")

    head = "".join(f"<th>@{r:.0%} cert. cov</th>" for r in report.target_risks)
    rows = []
    for e in report.entries:
        cells = [f"<td>{_esc(e.name)}</td>", f"<td>{_fmt(e.accuracy)}</td>", f"<td>{_fmt(e.ece_calibrated)}</td>"]
        for r in report.target_risks:
            key = rk(r)
            if not e.certified_feasible_by_risk.get(key):
                cells.append("<td>infeasible</td>")
            else:
                cells.append(f"<td>{_pct(e.certified_coverage_by_risk.get(key))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    winners = "".join(
        f"<li>@{r:.0%}: {_esc(report.best_by_coverage_at[rk(r)])}</li>"
        if rk(r) in report.best_by_coverage_at
        else f"<li>@{r:.0%}: no feasible certificate</li>"
        for r in report.target_risks
    )
    notes = "".join(f"<li>{_esc(n)}</li>" for n in report.notes)
    return _page(
        "DecisionOps Comparison",
        f"""
        <h1>DecisionOps comparison</h1>
        <p class="sub">dataset={_esc(report.dataset)} n_test={report.split.n_test}
          seed={report.split.seed}. Ranking uses CRC certificates
          (joint automated error rate).</p>
        <table>
          <thead><tr><th>Model</th><th>Acc</th><th>ECE cal</th>{head}</tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        <h2>Winners</h2>
        <ul>{winners}</ul>
        <ul class="notes">{notes}</ul>
        """,
    )


def _cascade_html(report: CascadeReport) -> str:
    points = [p.model_dump(mode="json") for p in report.curve]
    chart = cost_miss_svg(
        points,
        title="Cost saved vs miss rate (test)",
        baseline_mean_cost=report.baseline_mean_cost,
    )
    # Sparse table rows
    n = len(report.curve)
    keep = {0, n - 1} if n else set()
    for i, p in enumerate(report.curve):
        hundredths = int(round(p.threshold * 100))
        on_grid = abs(p.threshold * 100 - hundredths) < 1e-9
        if on_grid and hundredths % 5 == 0:
            keep.add(i)
        if n and i % max(1, n // 12) == 0:
            keep.add(i)
    rows = []
    for i in sorted(keep):
        p = report.curve[i]
        sel = "—" if p.selective_risk is None else _fmt(p.selective_risk)
        frac = "—" if p.cost_saved_frac is None else _pct(p.cost_saved_frac)
        rows.append(
            "<tr>"
            f"<td>{_fmt(p.threshold, 2)}</td>"
            f"<td>{_pct(p.automation_rate)}</td>"
            f"<td>{_fmt(p.miss_rate)}</td>"
            f"<td>{sel}</td>"
            f"<td>{_fmt(p.mean_cost)}</td>"
            f"<td>{_fmt(p.cost_saved)}</td>"
            f"<td>{frac}</td>"
            f"<td>{p.n_auto}/{p.n_total}</td>"
            "</tr>"
        )
    notes = "".join(f"<li>{_esc(n)}</li>" for n in report.notes)
    who = " ".join(
        part
        for part in (
            f"provider={report.provider}" if report.provider else "",
            f"model={report.model}" if report.model else "",
        )
        if part
    )
    cname = _esc(report.contract.get("name", "?"))
    auto = _esc(report.contract.get("auto_choices", []))
    esc = _esc(report.contract.get("escalate_choices", []))
    return _page(
        "DecisionOps Cascade",
        f"""
        <h1>DecisionOps cascade (cost vs miss rate)</h1>
        <p class="sub">n_total={report.split.n_total}
          n_cal={report.split.n_calibration}
          n_test={report.split.n_test}
          seed={report.split.seed}
          {_esc(who)}</p>
        <p class="sub">contract={cname} auto={auto} escalate={esc}.
          Baseline always-escalate mean={_fmt(report.baseline_mean_cost)};
          always-auto mean={_fmt(report.always_auto_mean_cost)},
          miss={_fmt(report.always_auto_miss_rate)}.</p>
        <p class="hint">Escalation pricing is oracle. Thresholds are empirical —
          use <code>decisionops certify</code> for a CRC bound on miss rate.</p>
        <div class="charts">{chart}</div>
        <h2>Operating points</h2>
        <table>
          <thead><tr>
            <th>Threshold</th><th>Automation</th><th>Miss rate</th><th>Sel. risk</th>
            <th>Mean cost</th><th>Cost saved</th><th>Saved %</th><th>Auto / N</th>
          </tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        <h2>Notes</h2>
        <ul class="notes">{notes}</ul>
        """,
    )



def _render(result: Report) -> str:
    if isinstance(result, GateReport):
        return _gate_html(result)
    if isinstance(result, EvalResult):
        return _eval_html(result)
    if isinstance(result, CertificationReport):
        return _cert_html(result)
    if isinstance(result, ComparisonReport):
        return _compare_html(result)
    if isinstance(result, CascadeReport):
        return _cascade_html(result)
    raise TypeError(f"unsupported report type: {type(result)!r}")
