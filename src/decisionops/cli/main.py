"""CLI entrypoint: eval | gate | compare | certify | cascade | decide | recertify | drift."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markup import escape

from decisionops import __version__
from decisionops.adapters import StaticFileProvider
from decisionops.cascade import run_cascade
from decisionops.compare import compare_models, load_prediction_sets
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract, load_contract
from decisionops.decide import (
    ProfileMismatchError,
    decide_many,
    load_profile,
    profile_from_report,
    save_profile,
)
from decisionops.drift import run_drift_monitor
from decisionops.eval import evaluate, run_gate
from decisionops.io import (
    load_dataset,
    load_decisions,
    load_outcomes,
    load_predictions,
)
from decisionops.recertify import run_recertification
from decisionops.report import (
    print_cascade_report,
    print_certification_report,
    print_comparison_report,
    print_drift_report,
    print_eval_report,
    print_gate_report,
    print_recertify_report,
    write_html_report,
    write_json_report,
)

if TYPE_CHECKING:
    from decisionops.cascade import CascadeReport
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


_PROVIDERS = ("static", "laya", "jev", "http")

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


NoBuiltinTemperatureOption = Annotated[
    bool,
    typer.Option(
        "--no-builtin-temperature",
        help="Laya only: ignore the checkpoint's bundled temperature table "
        "(often stale after fine-tuning).",
    ),
]

JevModelOption = Annotated[
    str | None,
    typer.Option(
        "--jev-model",
        help="Jev only: model id (default jev-latest); also a fine-tuned model id.",
    ),
]

JevQuestionOption = Annotated[
    str | None,
    typer.Option(
        "--jev-question",
        help="Jev only: choice (default) or noul. noul asks a calibrated yes/no "
        "question and needs a contract with yes_choice.",
    ),
]

JevBaseUrlOption = Annotated[
    str | None,
    typer.Option(
        "--jev-base-url",
        help="Jev only: API base URL for a private or custom deployment.",
    ),
]

HttpUrlOption = Annotated[
    str | None,
    typer.Option(
        "--http-url",
        help="HTTP provider: POST JSON {id, state, question, choices} to this URL.",
    ),
]

HttpTokenOption = Annotated[
    str | None,
    typer.Option(
        "--http-token",
        help="HTTP provider: Bearer token (else DECISIONOPS_HTTP_TOKEN).",
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
    no_builtin_temperature: bool = False,
    jev_model: str | None = None,
    jev_base_url: str | None = None,
    jev_question: str | None = None,
    http_url: str | None = None,
    http_token: str | None = None,
    contract: DecisionContract | None = None,
):
    """
    Resolve a prediction source.

    `--predictions` (with or without `--provider static`) is the offline path.
    `--provider laya|jev|http` runs live inference; `--predictions` is then rejected.
    A contract's descriptions / instructions become the live model's input, so the
    certified input and the production input are the same.
    """
    cases = load_dataset(dataset)
    if limit is not None:
        if limit < 1:
            raise typer.BadParameter("--limit must be >= 1")
        cases = cases[:limit]

    name = (provider_name or ("static" if predictions is not None else None))
    if name is None:
        raise typer.BadParameter("provide --predictions FILE or --provider laya|jev|http")
    name = name.strip().lower()
    if name not in _PROVIDERS:
        raise typer.BadParameter(
            f"unknown provider {provider_name!r}; choose from {', '.join(_PROVIDERS)}"
        )

    # Provider-specific flags must not be silently ignored on another provider.
    misplaced = [
        flag
        for flag, value, owner in (
            ("--model-dir", model_dir is not None, "laya"),
            ("--no-builtin-temperature", no_builtin_temperature, "laya"),
            ("--jev-model", jev_model is not None, "jev"),
            ("--jev-base-url", jev_base_url is not None, "jev"),
            ("--jev-question", jev_question is not None, "jev"),
            ("--http-url", http_url is not None, "http"),
            ("--http-token", http_token is not None, "http"),
        )
        if value and name != owner
    ]
    if misplaced:
        raise typer.BadParameter(
            f"{', '.join(misplaced)} not valid with --provider {name}"
        )

    if name == "static":
        if predictions is None:
            raise typer.BadParameter("--provider static requires --predictions")
        provider = StaticFileProvider(str(predictions))
        return cases, provider.predict_many(cases)

    if predictions is not None:
        raise typer.BadParameter("--predictions is only valid with --provider static")

    descriptions = contract.descriptions if contract is not None else None
    contract_instructions = contract.instructions if contract is not None else None

    if name == "laya":
        from decisionops.adapters.laya import LayaProvider

        kwargs: dict = {
            "apply_builtin_temperature": not no_builtin_temperature,
            "descriptions": descriptions,
            "contract_instructions": contract_instructions,
        }
        if model_dir is not None:
            kwargs["model_dir"] = model_dir
        provider = LayaProvider(**kwargs)
        console.print(
            f"[dim]Running Laya on {len(cases)} cases "
            f"(device={provider.device})…[/dim]"
        )
        return cases, provider.predict_many(cases)

    if name == "jev":
        from decisionops.adapters.jev import DEFAULT_MODEL, QUESTION_TYPES, JevProvider

        question_type = (jev_question or "choice").strip().lower()
        if question_type not in QUESTION_TYPES:
            raise typer.BadParameter(
                f"--jev-question must be one of {', '.join(QUESTION_TYPES)}"
            )
        yes_choice = contract.yes_choice if contract is not None else None
        if question_type == "noul" and yes_choice is None:
            raise typer.BadParameter(
                "--jev-question noul needs --contract with yes_choice "
                "(the choice that means 'yes')"
            )
        provider = JevProvider(
            model=jev_model or DEFAULT_MODEL,
            base_url=jev_base_url,
            descriptions=descriptions,
            contract_instructions=contract_instructions,
            question_type=question_type,
            yes_choice=yes_choice,
        )
        console.print(
            f"[dim]Running Jev ({provider.model}) on {len(cases)} cases…[/dim]"
        )
        try:
            return cases, provider.predict_many(cases)
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    if name == "http":
        from decisionops.adapters.http import HttpProvider

        if not http_url:
            raise typer.BadParameter("--provider http requires --http-url")
        try:
            provider = HttpProvider(
                http_url,
                token=http_token,
                descriptions=descriptions,
                contract_instructions=contract_instructions,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(
            f"[dim]Running HTTP provider ({provider.url}) on {len(cases)} cases…[/dim]"
        )
        try:
            return cases, provider.predict_many(cases)
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    raise typer.BadParameter(f"unhandled provider {name!r}")


def _write_reports(
    result: EvalResult | GateReport | ComparisonReport | CertificationReport | CascadeReport,
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
        typer.Option("--provider", help="Prediction source: static | laya | jev | http."),
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
    no_builtin_temperature: NoBuiltinTemperatureOption = False,
    jev_model: JevModelOption = None,
    jev_base_url: JevBaseUrlOption = None,
    jev_question: JevQuestionOption = None,
    http_url: HttpUrlOption = None,
    http_token: HttpTokenOption = None,
    contract: ContractOption = None,
) -> None:
    """
    Quick same-set evaluation (smoke / debugging).

    For serious reliability decisions use `decisionops gate` (cal/test split).

    Examples:
      decisionops eval data.jsonl -p preds.jsonl
      decisionops eval data.jsonl --provider laya --limit 50
      decisionops eval data.jsonl --provider jev --limit 20
      decisionops eval data.jsonl --provider http --http-url https://model.example/v1/predict
    """
    if n_bins < 1:
        raise typer.BadParameter("--n-bins must be >= 1")

    risks = _parse_risks(target_risks)
    decision_contract = _load_contract_option(contract)
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
        no_builtin_temperature=no_builtin_temperature,
        jev_model=jev_model,
        jev_base_url=jev_base_url,
        jev_question=jev_question,
        http_url=http_url,
        http_token=http_token,
        contract=decision_contract,
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
        typer.Option("--provider", help="Prediction source: static | laya | jev | http."),
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
    no_builtin_temperature: NoBuiltinTemperatureOption = False,
    jev_model: JevModelOption = None,
    jev_base_url: JevBaseUrlOption = None,
    jev_question: JevQuestionOption = None,
    http_url: HttpUrlOption = None,
    http_token: HttpTokenOption = None,
    verdict_alpha: Annotated[
        float,
        typer.Option(
            "--verdict-alpha",
            help="Joint automated error rate the go/watch/no-go verdict is certified at.",
        ),
    ] = 0.05,
    min_coverage: Annotated[
        float,
        typer.Option(
            "--min-coverage",
            help="Certified test coverage needed for 'go' (share of traffic automated). "
            "A policy choice: set it from your cost of escalation.",
        ),
    ] = 0.30,
) -> None:
    """
    Technical gate: cal/test split, optional temperature + isotonic, held-out metrics.

    The verdict comes from a CRC certificate at --verdict-alpha: no-go if none
    exists, watch if its test coverage is below --min-coverage, else go. ECE is
    reported for context and never decides the verdict.

    Examples:
      decisionops gate data.jsonl -p preds.jsonl
      decisionops gate data.jsonl --provider laya
      decisionops gate data.jsonl --provider jev --contract contract.json
    """
    if n_bins < 1:
        raise typer.BadParameter("--n-bins must be >= 1")
    if not 0.0 < cal_ratio < 1.0:
        raise typer.BadParameter("--cal-ratio must be in (0, 1)")
    if not 0.0 < verdict_alpha < 1.0:
        raise typer.BadParameter("--verdict-alpha must be in (0, 1)")
    if not 0.0 <= min_coverage <= 1.0:
        raise typer.BadParameter("--min-coverage must be in [0, 1]")

    risks = _parse_risks(target_risks)
    decision_contract = _load_contract_option(contract)
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
        no_builtin_temperature=no_builtin_temperature,
        jev_model=jev_model,
        jev_base_url=jev_base_url,
        jev_question=jev_question,
        http_url=http_url,
        http_token=http_token,
        contract=decision_contract,
    )
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
            verdict_alpha=verdict_alpha,
            min_coverage=min_coverage,
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
        typer.Option("--provider", help="Prediction source: static | laya | jev | http."),
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
    no_builtin_temperature: NoBuiltinTemperatureOption = False,
    jev_model: JevModelOption = None,
    jev_base_url: JevBaseUrlOption = None,
    jev_question: JevQuestionOption = None,
    http_url: HttpUrlOption = None,
    http_token: HttpTokenOption = None,
    profile: Annotated[
        Path | None,
        typer.Option(
            "--profile",
            help="Write a runtime CertificationProfile JSON for --profile-alpha.",
            dir_okay=False,
        ),
    ] = None,
    profile_alpha: Annotated[
        float | None,
        typer.Option(
            "--profile-alpha",
            help="Alpha to freeze into --profile (must be one of --alphas). "
            "Defaults to the only alpha when exactly one is given.",
        ),
    ] = None,
    confidence_level: Annotated[
        float | None,
        typer.Option(
            "--confidence-level",
            help="e.g. 0.95: high-probability certificate (P(joint error <= alpha) "
            ">= this) instead of CRC's in-expectation bound. Needs more data.",
        ),
    ] = None,
    audit_rate: Annotated[
        float,
        typer.Option(
            "--audit-rate",
            help="Profile: share of traffic decide() forces to the expensive path "
            "as an unbiased audit sample for recertification (e.g. 0.05).",
        ),
    ] = 0.0,
    audit_seed: Annotated[
        int,
        typer.Option("--audit-seed", help="Profile: salt for the audit draw."),
    ] = 0,
    risk: Annotated[
        str,
        typer.Option(
            "--risk",
            help="joint: share of all traffic auto-accepted AND wrong (CRC). "
            "selective: error rate AMONG auto-accepted cases (Learn-Then-Test; "
            "high-probability only, --confidence-level defaults to 0.9).",
        ),
    ] = "joint",
) -> None:
    """
    Certified abstention thresholds via Conformal Risk Control.

    Unlike `gate`, the thresholds here carry a finite-sample guarantee: the
    **joint automated error rate** (share of all traffic auto-accepted AND wrong)
    is bounded by alpha in expectation, under exchangeability. This is not the
    conditional selective risk that `gate` reports empirically.

    With --confidence-level the bound instead holds with that probability over
    the calibration draw (Clopper-Pearson RCPS).

    With --risk selective the bound is on the error rate among auto-accepted
    cases ("of what you automate, at most alpha is wrong").

    Levels with too little calibration data, or where no threshold can hold
    alpha, are reported as infeasible rather than given a threshold.
    """
    if not 0.0 < cal_ratio < 1.0:
        raise typer.BadParameter("--cal-ratio must be in (0, 1)")
    if not 0.0 <= temp_ratio < 1.0:
        raise typer.BadParameter("--temp-ratio must be in [0, 1)")
    if confidence_level is not None and not 0.0 < confidence_level < 1.0:
        raise typer.BadParameter("--confidence-level must be in (0, 1)")
    if not 0.0 <= audit_rate < 1.0:
        raise typer.BadParameter("--audit-rate must be in [0, 1)")
    if risk not in ("joint", "selective"):
        raise typer.BadParameter("--risk must be 'joint' or 'selective'")
    if profile is None and (audit_rate != 0.0 or audit_seed != 0):
        raise typer.BadParameter("--audit-rate / --audit-seed need --profile")

    levels = _parse_risks(alphas)
    if profile_alpha is not None and profile is None:
        raise typer.BadParameter("--profile-alpha needs --profile")
    if profile is not None and profile_alpha is None:
        if len(levels) != 1:
            raise typer.BadParameter("--profile with several --alphas needs --profile-alpha")
        profile_alpha = levels[0]
    if profile_alpha is not None and not any(abs(a - profile_alpha) < 1e-12 for a in levels):
        raise typer.BadParameter(f"--profile-alpha {profile_alpha:g} is not in --alphas")

    decision_contract = _load_contract_option(contract)
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
        no_builtin_temperature=no_builtin_temperature,
        jev_model=jev_model,
        jev_base_url=jev_base_url,
        jev_question=jev_question,
        http_url=http_url,
        http_token=http_token,
        contract=decision_contract,
    )
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
            confidence_level=confidence_level,
            risk=risk,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_certification_report(report, console=console)

    if output is not None:
        _write_reports(report, output, "certification_report")

    if profile is not None and profile_alpha is not None:
        try:
            frozen = profile_from_report(
                report, profile_alpha, audit_rate=audit_rate, audit_seed=audit_seed
            )
        except ValueError as exc:
            console.print(f"[red]No profile written:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        path = save_profile(frozen, profile)
        console.print(
            f"[green]Wrote[/green] {path}  "
            f"(alpha={frozen.alpha:g}, threshold={frozen.threshold:.4f}, "
            f"audit_rate={frozen.audit_rate:g}, profile={frozen.profile_hash})"
        )


@app.command("cascade")
def cascade_cmd(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to labeled dataset JSONL.", exists=True, dir_okay=False),
    ],
    contract: Annotated[
        Path,
        typer.Option(
            "--contract",
            help="Decision contract JSON with costs for every choice.",
            exists=True,
            dir_okay=False,
        ),
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
        typer.Option("--provider", help="Prediction source: static | laya | jev | http."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for JSON + HTML reports."),
    ] = None,
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local Laya snapshot (skips Hub download)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N cases."),
    ] = None,
    cal_ratio: Annotated[
        float,
        typer.Option("--cal-ratio", help="Calibration fraction (curve scored on test)."),
    ] = 0.3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for the split."),
    ] = 42,
    no_builtin_temperature: NoBuiltinTemperatureOption = False,
    jev_model: JevModelOption = None,
    jev_base_url: JevBaseUrlOption = None,
    jev_question: JevQuestionOption = None,
    http_url: HttpUrlOption = None,
    http_token: HttpTokenOption = None,
) -> None:
    """
    Offline cascade: cost saved vs automated miss rate.

    Needs a contract with costs. Escalation is priced with an oracle (true
    escalate label, else cheapest escalate choice). Thresholds are empirical —
    pair with `certify` for a CRC bound on the miss rate.
    """
    if not 0.0 < cal_ratio < 1.0:
        raise typer.BadParameter("--cal-ratio must be in (0, 1)")

    decision_contract = _load_contract_option(contract)
    if decision_contract is None:
        raise typer.BadParameter("--contract is required")
    cases, preds = _load_cases_and_predictions(
        dataset,
        predictions=predictions,
        provider_name=provider,
        model_dir=model_dir,
        limit=limit,
        no_builtin_temperature=no_builtin_temperature,
        jev_model=jev_model,
        jev_base_url=jev_base_url,
        jev_question=jev_question,
        http_url=http_url,
        http_token=http_token,
        contract=decision_contract,
    )
    try:
        report = run_cascade(
            cases,
            preds,
            decision_contract,
            cal_ratio=cal_ratio,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_cascade_report(report, console=console)

    if output is not None:
        _write_reports(report, output, "cascade_report")


@app.command("decide")
def decide_cmd(
    profile: Annotated[
        Path,
        typer.Argument(
            help="CertificationProfile JSON from `certify --profile`.",
            exists=True,
            dir_okay=False,
        ),
    ],
    predictions: Annotated[
        Path,
        typer.Option(
            "--predictions",
            "-p",
            help="Predictions JSONL to route (raw probabilities; T is applied here).",
            exists=True,
            dir_okay=False,
        ),
    ],
    contract: Annotated[
        Path | None,
        typer.Option(
            "--contract",
            help="Contract in use now; rejected unless it matches the certified one.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write one Decision per line (JSONL)."),
    ] = None,
) -> None:
    """
    Route predictions with a certified profile: AUTO or ESCALATE.

    AUTO only when the predicted choice is automatic and calibrated confidence
    is at or above the certified threshold; everything else escalates. Offline
    smoke for the runtime rule — the real consumer is the library `decide()`.
    """
    try:
        frozen = load_profile(profile)
        frozen.check_contract(_load_contract_option(contract))
        decisions = decide_many(load_predictions(predictions), frozen)
    except (ProfileMismatchError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    n = len(decisions)
    n_auto = sum(d.action == "auto" for d in decisions)
    n_audit = sum(d.audit for d in decisions)
    id_counts = Counter(d.id for d in decisions)
    duplicated = sorted(i for i, c in id_counts.items() if c > 1)
    by_reason: dict[str, int] = {}
    for d in decisions:
        by_reason[d.reason] = by_reason.get(d.reason, 0) + 1

    console.print()
    console.rule("[bold]DecisionOps Decide[/bold]")
    console.print(
        f"profile={frozen.profile_hash}  alpha={frozen.alpha:g}  "
        f"threshold={frozen.threshold:.4f}  "
        + (f"T={frozen.temperature:.4f}" if frozen.temperature is not None else "T=none")
    )
    console.print(
        f"guarantee={frozen.guarantee}"
        + (
            f" (confidence_level={frozen.confidence_level:g})"
            if frozen.confidence_level is not None
            else ""
        )
    )
    if frozen.model or frozen.provider:
        console.print(
            (f"provider={frozen.provider}  " if frozen.provider else "")
            + (f"model={frozen.model}" if frozen.model else "")
        )
    console.print(
        f"n={n}  auto={n_auto} ({n_auto / n:.1%})  escalate={n - n_auto}  "
        + "  ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
    )
    if frozen.audit_rate > 0.0:
        console.print(
            f"audit sample: {n_audit} ({n_audit / n:.1%}) at audit_rate="
            f"{frozen.audit_rate:g}, seed={frozen.audit_seed}; recertify on audit=true rows"
        )
    if duplicated:
        # The audit draw keys on id, so repeated ids share one draw: fine for
        # retries of the same case, wrong if distinct cases reuse an id.
        shown = ", ".join(repr(i) for i in duplicated[:5])
        more = f" (+{len(duplicated) - 5} more)" if len(duplicated) > 5 else ""
        console.print(
            f"[yellow]Warning:[/yellow] {len(duplicated)} id(s) appear more than once: "
            f"{escape(shown)}{more}. Repeated ids get the same audit draw; if they are "
            "distinct cases, give each a unique id or the audit sample is not uniform."
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(d.model_dump(mode="json"), ensure_ascii=False) for d in decisions]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")



@app.command("recertify")
def recertify_cmd(
    profile: Annotated[
        Path,
        typer.Argument(
            help="Current CertificationProfile JSON to re-check.",
            exists=True,
            dir_okay=False,
        ),
    ],
    decisions: Annotated[
        Path,
        typer.Option(
            "--decisions",
            help="Decisions JSONL from `decisionops decide -o` (needs audit flags).",
            exists=True,
            dir_okay=False,
        ),
    ],
    outcomes: Annotated[
        Path,
        typer.Option(
            "--outcomes",
            help="Outcome feedback JSONL: id, label, optional label_source / matured_at.",
            exists=True,
            dir_okay=False,
        ),
    ],
    predictions: Annotated[
        Path,
        typer.Option(
            "--predictions",
            "-p",
            help="Raw predictions JSONL for the same ids (T is applied from the profile).",
            exists=True,
            dir_okay=False,
        ),
    ],
    contract: Annotated[
        Path | None,
        typer.Option(
            "--contract",
            help="Contract in use now; must match the profile fingerprint.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help="ISO-8601: only outcomes with matured_at <= this time are used.",
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="ISO-8601: drop outcomes matured before this time (rolling window).",
        ),
    ] = None,
    confidence_level: Annotated[
        float | None,
        typer.Option(
            "--confidence-level",
            help="Override profile guarantee; default keeps the profile's confidence_level.",
        ),
    ] = None,
    profile_out: Annotated[
        Path | None,
        typer.Option(
            "--profile-out",
            help="Write the new CertificationProfile when the audit certificate is feasible.",
            dir_okay=False,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write RecertifyReport JSON."),
    ] = None,
) -> None:
    """
    Re-certify a live profile from random audit samples only.

    Joins decisions (audit=True) with outcome labels and raw predictions, applies
    the frozen temperature, and runs CRC at the profile alpha. Organic escalations
    are ignored. Writes a replacement profile when feasible and --profile-out is set.
    """
    if confidence_level is not None and not 0.0 < confidence_level < 1.0:
        raise typer.BadParameter("--confidence-level must be in (0, 1)")
    try:
        frozen = load_profile(profile)
        report = run_recertification(
            frozen,
            decisions=load_decisions(decisions),
            outcomes=load_outcomes(outcomes),
            predictions=load_predictions(predictions),
            contract=_load_contract_option(contract),
            as_of=as_of,
            since=since,
            confidence_level=confidence_level,
        )
    except (ProfileMismatchError, ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    print_recertify_report(report, console=console)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Drop the embedded profile from the JSON dump path via write helper shape
        payload = report.model_dump(mode="json")
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/green] {output}")

    if profile_out is not None:
        if report.profile is None:
            console.print(
                "[red]No profile written:[/red] audit certificate is infeasible."
            )
            raise typer.Exit(code=1)
        path = save_profile(report.profile, profile_out)
        console.print(
            f"[green]Wrote[/green] {path}  "
            f"(verdict={report.verdict}, threshold={report.profile.threshold:.4f}, "
            f"profile={report.profile.profile_hash})"
        )
    elif report.verdict == "infeasible":
        raise typer.Exit(code=1)




@app.command("drift")
def drift_cmd(
    live: Annotated[
        Path,
        typer.Argument(
            help="Live decisions JSONL from `decisionops decide -o`.",
            exists=True,
            dir_okay=False,
        ),
    ],
    baseline: Annotated[
        Path,
        typer.Option(
            "--baseline",
            help="Baseline decisions JSONL from the same profile (reference window).",
            exists=True,
            dir_okay=False,
        ),
    ],
    profile: Annotated[
        Path | None,
        typer.Option(
            "--profile",
            help="Optional CertificationProfile for hash/coverage context.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    min_window: Annotated[
        int,
        typer.Option("--min-window", help="Minimum decisions per window."),
    ] = 30,
    bins: Annotated[
        int,
        typer.Option("--bins", help="Equal-width bins for confidence PSI."),
    ] = 10,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write DriftReport JSON."),
    ] = None,
) -> None:
    """
    Label-free drift monitor: compare live decide() behaviour to a baseline.

    Watches certified rate (reason in {certified, audit}) and confidence
    distribution (PSI + KS). A shift suggests raising audit_rate or running
    recertify — it is not itself a certificate. Exit 2 on recertify_suggested,
    1 on watch, 0 on ok.
    """
    if min_window < 2:
        raise typer.BadParameter("--min-window must be >= 2")
    if bins < 2:
        raise typer.BadParameter("--bins must be >= 2")
    try:
        frozen = load_profile(profile) if profile is not None else None
        report = run_drift_monitor(
            load_decisions(live),
            load_decisions(baseline),
            profile=frozen,
            n_bins=bins,
            min_window=min_window,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    print_drift_report(report, console=console)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/green] {output}")

    if report.verdict == "recertify_suggested":
        raise typer.Exit(code=2)
    if report.verdict == "watch":
        raise typer.Exit(code=1)



if __name__ == "__main__":
    app()
