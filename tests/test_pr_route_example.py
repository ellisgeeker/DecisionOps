"""The GitHub Action example: PR event -> case -> certified decide -> outputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from decisionops.decide import load_profile
from decisionops.io import load_dataset
from decisionops.schemas import DecisionPrediction

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/github-action/pr_route.py"
PROFILE = ROOT / "examples/code-review/default/profile.json"
STRICT = ROOT / "examples/code-review/strict/profile.json"
CONTRACT = ROOT / "examples/contracts/code-review-depth.json"
STRICT_CONTRACT = ROOT / "examples/contracts/code-review-depth-strict.json"
EVENT = ROOT / "examples/github-action/sample_event.json"
LEVELS = ["skip", "shallow", "deep", "human"]


def _module():
    spec = importlib.util.spec_from_file_location("pr_route", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeJev:
    """Stands in for JevProvider: records cases, returns a fixed distribution."""

    def __init__(self, probs, model="jev-1.14.0"):
        self.probs = probs
        self.model = model
        self.cases = []

    def predict(self, case):
        self.cases.append(case)
        top = max(self.probs, key=self.probs.get)
        return DecisionPrediction(
            id=case.id, prediction=top, probabilities=dict(self.probs), model=self.model
        )


def _pr(**over):
    pr = {
        "number": 7,
        "title": "Fix typo in auth check",
        "body": "`is_admin` was misspelled.\n\nSecond paragraph.",
        "additions": 1,
        "deletions": 1,
        "user": {"login": "dev-a"},
        "head": {"sha": "abc"},
        "labels": [],
    }
    return pr | over


def test_case_layout_matches_the_certified_pack():
    """A certificate holds for the input it saw: same `state` lines as the pack."""
    mod = _module()
    case = mod.case_from_pr(_pr(), ["api/auth/permissions.py"], LEVELS)
    pack = load_dataset(ROOT / "examples/code-review/dataset.jsonl")[0]
    keys = [line.split(":", 1)[0] for line in case.state.splitlines()]
    assert keys == [line.split(":", 1)[0] for line in pack.state.splitlines()]
    assert case.question == pack.question
    assert case.choices == pack.choices
    # Multi-line bodies are flattened, as in the pack.
    assert "Description: `is_admin` was misspelled. Second paragraph." in case.state


def test_case_truncates_long_inputs():
    mod = _module()
    files = [f"f{i}.py" for i in range(55)]
    case = mod.case_from_pr(_pr(body="x" * 5000), files, LEVELS)
    assert "(+15 more)" in case.state
    assert "Files changed (55)" in case.state
    desc = next(line for line in case.state.splitlines() if line.startswith("Description"))
    assert len(desc) < 700


def test_route_auto_and_escalate_with_real_profiles():
    mod = _module()
    contract = mod.load_contract(CONTRACT)
    case = mod.case_from_pr(_pr(), ["api/auth/permissions.py"], LEVELS)

    profile = load_profile(PROFILE)
    confident = FakeJev({"skip": 0.0, "shallow": 0.0, "deep": 0.02, "human": 0.98})
    pred, d = mod.route(case, profile, contract, confident)
    assert d.action == "auto" and d.choice == "human"
    assert mod.route_label(d) == "review:human"

    strict = load_profile(STRICT)
    unsure = FakeJev({"skip": 0.4, "shallow": 0.35, "deep": 0.2, "human": 0.05})
    _, d = mod.route(case, strict, mod.load_contract(STRICT_CONTRACT), unsure)
    assert d.action == "escalate" and d.reason == "below_threshold"
    assert mod.route_label(d) == "review:escalate"


def test_route_refuses_wrong_contract_or_model():
    mod = _module()
    case = mod.case_from_pr(_pr(), ["a.py"], LEVELS)
    profile = load_profile(PROFILE)
    with pytest.raises(ValueError, match="differs from the certified"):
        mod.route(case, profile, mod.load_contract(STRICT_CONTRACT), FakeJev({"skip": 1.0}))
    with pytest.raises(ValueError, match="profile certifies"):
        other = FakeJev({"skip": 0.9, "deep": 0.1}, model="jev-9.9")
        mod.route(case, profile, mod.load_contract(CONTRACT), other)


def test_jev_provider_is_pinned_to_the_certified_model(monkeypatch):
    pytest.importorskip("typesafe_sdk")
    mod = _module()
    from decisionops.adapters import jev as jev_mod

    seen = {}

    class _Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(jev_mod._require_sdk(), "TypeSafeClient", _Client)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    provider = mod.jev_provider(load_profile(PROFILE), mod.load_contract(CONTRACT))
    assert seen["model"] == "jev-1.14.0"
    assert provider.question_type == "score"
    assert provider.choice_order == LEVELS


def test_main_dry_run_writes_outputs_summary_and_record(tmp_path, monkeypatch):
    mod = _module()
    out, summary = tmp_path / "out", tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    record = tmp_path / "decision.jsonl"
    jev = FakeJev({"skip": 0.0, "shallow": 0.0, "deep": 0.03, "human": 0.97})
    rc = mod.main(
        ["--event", str(EVENT), "--profile", str(PROFILE), "--contract", str(CONTRACT),
         "--decisions-out", str(record), "--no-label"],
        provider=jev,
    )
    assert rc == 0
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["action"] == "auto"
    assert outputs["level"] == "human"
    assert outputs["label"] == "review:human"
    assert outputs["audit"] in {"true", "false"}
    assert "DecisionOps route for #101" in summary.read_text()
    row = json.loads(record.read_text())
    assert row["id"] == "pr-101" and row["model"] == "jev-1.14.0"
    assert row["profile"] == load_profile(PROFILE).profile_hash
    # The PR text reached the model only as `state`.
    assert "Fix typo in auth check" in jev.cases[0].state


def test_label_replacement_removes_stale_route_labels(monkeypatch):
    mod = _module()
    calls = []
    gh = mod.GitHub("o/r", "t")
    monkeypatch.setattr(gh, "_call", lambda m, p, b=None: calls.append((m, p, b)))
    gh.set_route_label(5, "review:deep", ["bug", "review:skip", "review:deep"])
    assert calls == [("DELETE", "/repos/o/r/issues/5/labels/review%3Askip", None)]

    calls.clear()
    gh.set_route_label(5, "review:escalate", ["bug"])
    assert calls == [("POST", "/repos/o/r/issues/5/labels", {"labels": ["review:escalate"]})]
