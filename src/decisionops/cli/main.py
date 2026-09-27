"""CLI entrypoint: decisionops eval | gate | compare | certify ..."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from decisionops import __version__
from decisionops.adapters import StaticFileProvider
from decisionops.compare import compare_models, load_prediction_sets
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract, load_contract
from decisionops.eval import evaluate, run_gate
from decisionops.io import load_dataset
from decisionops.report import (
    print_certification_report,
    print_comparison_report,
    print_eval_report,
    print_gate_report,
    write_html_report,
    write_json_report,
)

if TYPE_CHECKING:
    from decisionops.conformal.certify import CertificationReport
    from decisionops.schemas import ComparisonReport, EvalResult, GateReport

app = typer.Typer(
    name="decisionops",
    help="Reliability infrastructure for AI decision models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _parse_risks(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("at least one target risk is required")
    risks: list[float] = []
    for p in parts:
        try:
            r = float(p)
        except ValueError as exc:
            raise typer.BadParameter(f"invalid risk value: {p!r}") from exc
        if not 0.0 < r < 1.0:
            raise typer.BadParameter(f"target risk must be in (0, 1), got {r}")
        risks.append(r)
    return risks


_PROVIDERS = ("static", "laya")

ContractOption = Annotated[
    Path | None,
    typer.Option(
        "--contract",
        help="Decision contract JSON. Only its auto_choices count as risk; "
        "escalations never do.",
        exists=True,
        dir_okay=False,
    ),
]


def _load_contract_option(path: Path | None) -> DecisionContract | None:
    if path is None:
        return None
    try:
        return load_contract(path)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid contract {path}: {exc}") from exc


def _load_cases_and_predictions(
    dataset: Path,
    *,
    predictions: Path | None,
    provider_name: str | None,
    model_dir: Path | None = None,
    limit: int | None = None,
):
    """
    Resolve a prediction source.

    `--predictions` (with or without `--provider static`) is the offline path.
    `--provider laya` runs local inference; `--predictions` is then rejected.
    """
    cases = load_dataset(dataset)
    if limit is not None:
        if limit < 1:
            raise typer.BadParameter("--limit must be >= 1")
        cases = cases[:limit]

    name = (provider_name or ("static" if predictions is not None else None))
    if name is None:
        raise typer.BadParameter("provide --predictions FILE or --provider laya")
    name = name.strip().lower()
    if name not in _PROVIDERS:
        raise typer.BadParameter(
            f"unknown provider {provider_name!r}; choose from {', '.join(_PROVIDERS)}"
        )

    if name == "static":
        if predictions is None:
            raise typer.BadParameter("--provider static requires --predictions")
        provider = StaticFileProvider(str(predictions))
        return cases, provider.predict_many(cases)

    if predictions is not None:
        raise typer.BadParameter("--predictions is only valid with --provider static")

    if name == "laya":
        from decisionops.adapters.laya import LayaProvider

        kwargs: dict = {}
        if model_dir is not None:
            kwargs["model_dir"] = model_dir
        provider = LayaProvider(**kwargs)
        console.print(
            f"[dim]Running Laya on {len(cases)} cases "
            f"(device={provider.device})…[/dim]"
        )
        return cases, provider.predict_many(cases)

    raise typer.BadParameter(f"unhandled provider {name!r}")


def _write_reports(
    result: EvalResult | GateReport | ComparisonReport | CertificationReport,
    out_dir: Path,
    stem: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json_report(result, out_dir / f"{stem}.json")
    html_path = write_html_report(result, out_dir / f"{stem}.html")
    console.print(f"[green]Wrote[/green] {json_path}")
    console.print(f"[green]Wrote[/green] {html_path}")


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", help="Show version and exit.", is_eager=True),
    ] = None,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command("eval")
def eval_cmd(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to labeled dataset JSONL.", exists=True, dir_okay=False),
    ],
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            "-p",
            help="Precomputed predictions JSONL (implies --provider static).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Prediction source: static | laya."),
    ] = None,
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local Laya snapshot (skips Hub download)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N cases."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for JSON + HTML reports."),
    ] = None,
    target_risks: Annotated[
        str,
        typer.Option(
            "--target-risks",
            help="Comma-separated target risks, e.g. 0.01,0.05,0.10",
        ),
    ] = "0.01,0.05,0.10",
    n_bins: Annotated[
        int,
        typer.Option("--n-bins", help="Number of fixed bins for ECE / reliability diagram."),
    ] = 10,
) -> None:
    """
    Quick same-set evaluation (smoke / debugging).

    For serious reliability decisions use `decisionops gate` (cal/test split).

    Examples:
      decisionops eval data.jsonl -p preds.jsonl
      decisionops eval data.jsonl --provider laya --limit 50
    """
    if n_bins < 1:
        raise typer.BadParameter("--n-bins must be >= 1")

    risks = _parse_risks(target_risks)
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
    )
    result = evaluate(
        cases,
        preds,
        n_bins=n_bins,
        target_risks=risks,
    )
    print_eval_report(result, console=console)

    if output is not None:
        _write_reports(result, output, "report")


@app.command("gate")
def gate_cmd(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to labeled dataset JSONL.", exists=True, dir_okay=False),
    ],
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            "-p",
            help="Precomputed predictions JSONL (implies --provider static).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Prediction source: static | laya."),
    ] = None,
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local Laya snapshot (skips Hub download)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N cases."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for JSON + HTML reports."),
    ] = None,
    cal_ratio: Annotated[
        float,
        typer.Option("--cal-ratio", help="Fraction of data used for calibration."),
    ] = 0.3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for the cal/test split."),
    ] = 42,
    target_risks: Annotated[
        str,
        typer.Option("--target-risks", help="Comma-separated target risks."),
    ] = "0.01,0.05,0.10",
    n_bins: Annotated[
        int,
        typer.Option("--n-bins", help="Bins for ECE / reliability diagram."),
    ] = 10,
    calibrate: Annotated[
        bool,
        typer.Option(
            "--calibrate/--no-calibrate",
            help="Fit temperature scaling and isotonic regression on cal split.",
        ),
    ] = True,
    contract: ContractOption = None,
) -> None:
    """
    Technical gate: cal/test split, optional temperature + isotonic, held-out metrics.

    Examples:
      decisionops gate data.jsonl -p preds.jsonl
      decisionops gate data.jsonl --provider laya
    """
    if n_bins < 1:
        raise typer.BadParameter("--n-bins must be >= 1")
    if not 0.0 < cal_ratio < 1.0:
        raise typer.BadParameter("--cal-ratio must be in (0, 1)")

    risks = _parse_risks(target_risks)
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
    )
    decision_contract = _load_contract_option(contract)
    try:
        report = run_gate(
            cases,
            preds,
            cal_ratio=cal_ratio,
            seed=seed,
            n_bins=n_bins,
            target_risks=risks,
            calibrate=calibrate,
            contract=decision_contract,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_gate_report(report, console=console)

    if output is not None:
        _write_reports(report, output, "gate_report")


@app.command("compare")
def compare_cmd(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to labeled dataset JSONL.", exists=True, dir_okay=False),
    ],
    predictions: Annotated[
        list[str],
        typer.Option(
            "--predictions",
            "-p",
            help="Repeatable. Use name=path to label a model, e.g. tfidf=preds.jsonl",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for JSON + HTML reports."),
    ] = None,
    cal_ratio: Annotated[
        float,
        typer.Option("--cal-ratio", help="Calibration fraction (shared across models)."),
    ] = 0.3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Split seed (shared across models)."),
    ] = 42,
    target_risks: Annotated[
        str,
        typer.Option("--target-risks", help="Comma-separated target risks."),
    ] = "0.01,0.05,0.10",
    n_bins: Annotated[
        int,
        typer.Option("--n-bins", help="Bins for ECE."),
    ] = 10,
    calibrate: Annotated[
        bool,
        typer.Option("--calibrate/--no-calibrate", help="Fit temperature per model."),
    ] = True,
    temp_ratio: Annotated[
        float,
        typer.Option(
            "--temp-ratio",
            help="Share of the calibration pool carved out to fit temperature "
            "(CRC set is never used for the fit).",
        ),
    ] = 0.2,
    contract: ContractOption = None,
) -> None:
    """
    Compare several prediction sets on one dataset under an identical split.

    Rankings use CRC certificates (joint automated error rate). Empirical
    thresholds are printed for contrast and marked ! when they rebound on test.
    """
    if len(predictions) < 2:
        raise typer.BadParameter("provide at least two --predictions to compare")
    if not 0.0 < cal_ratio < 1.0:
        raise typer.BadParameter("--cal-ratio must be in (0, 1)")
    if not 0.0 <= temp_ratio < 1.0:
        raise typer.BadParameter("--temp-ratio must be in [0, 1)")

    risks = _parse_risks(target_risks)
    cases = load_dataset(dataset)
    try:
        sets = load_prediction_sets(list(predictions))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    decision_contract = _load_contract_option(contract)
    try:
        report = compare_models(
            cases,
            sets,
            dataset_name=dataset.parent.name or dataset.stem,
            cal_ratio=cal_ratio,
            seed=seed,
            n_bins=n_bins,
            target_risks=risks,
            calibrate=calibrate,
            temp_ratio=temp_ratio,
            contract=decision_contract,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_comparison_report(report, console=console)

    if output is not None:
        _write_reports(report, output, "compare_report")


@app.command("certify")
def certify_cmd(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to labeled dataset JSONL.", exists=True, dir_okay=False),
    ],
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            "-p",
            help="Precomputed predictions JSONL (implies --provider static).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Prediction source: static | laya."),
    ] = None,
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local Laya snapshot (skips Hub download)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N cases."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for JSON + HTML reports."),
    ] = None,
    cal_ratio: Annotated[
        float,
        typer.Option("--cal-ratio", help="Fraction of data held back from test."),
    ] = 0.3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for the split."),
    ] = 42,
    alphas: Annotated[
        str,
        typer.Option(
            "--alphas",
            help="Comma-separated target bounds on the joint automated error rate.",
        ),
    ] = "0.01,0.05,0.10",
    temp_ratio: Annotated[
        float,
        typer.Option(
            "--temp-ratio",
            help="Share of the calibration pool carved out to fit temperature. "
            "0 skips the fit; the CRC set is never used for it.",
        ),
    ] = 0.2,
    calibrate: Annotated[
        bool,
        typer.Option("--calibrate/--no-calibrate", help="Fit temperature scaling."),
    ] = True,
    contract: ContractOption = None,
) -> None:
    """
    Certified abstention thresholds via Conformal Risk Control.

    Unlike `gate`, the thresholds here carry a finite-sample guarantee: the
    **joint automated error rate** (share of all traffic auto-accepted AND wrong)
    is bounded by alpha in expectation, under exchangeability. This is not the
    conditional selective risk that `gate` reports empirically.

    Levels with too little calibration data, or where no threshold can hold
    alpha, are reported as infeasible rather than given a threshold.
    """
    if not 0.0 < cal_ratio < 1.0:
        raise typer.BadParameter("--cal-ratio must be in (0, 1)")
    if not 0.0 <= temp_ratio < 1.0:
        raise typer.BadParameter("--temp-ratio must be in [0, 1)")

    levels = _parse_risks(alphas)
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
    )
    decision_contract = _load_contract_option(contract)
    try:
        report = run_certification(
            cases,
            preds,
            alphas=levels,
            cal_ratio=cal_ratio,
            temp_ratio=temp_ratio,
            seed=seed,
            calibrate=calibrate,
            contract=decision_contract,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_certification_report(report, console=console)

    if output is not None:
        _write_reports(report, output, "certification_report")


if __name__ == "__main__":
    app()
