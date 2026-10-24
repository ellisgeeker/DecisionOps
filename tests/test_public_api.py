"""The top-level package is the supported API; this pins it."""

from __future__ import annotations

import numpy as np

import decisionops as dops


def test_all_names_resolve():
    for name in dops.__all__:
        assert hasattr(dops, name), name


def test_certify_profile_decide_from_top_level(tmp_path):
    contract = dops.DecisionContract(
        name="routing",
        choices=["skip", "deep"],
        auto_choices=["skip"],
        severity={"deep": 2.0},
    )
    rng = np.random.default_rng(0)
    cases, preds = [], []
    for i in range(600):
        conf = float(rng.uniform(0.5, 0.999))
        label = "skip" if rng.random() < conf**2 else "deep"
        cases.append(dops.DecisionCase(id=str(i), choices=["skip", "deep"], label=label))
        preds.append(
            dops.DecisionPrediction(
                id=str(i), prediction="skip", probabilities={"skip": conf, "deep": 1 - conf}
            )
        )

    report = dops.run_certification(cases, preds, alphas=[0.10], contract=contract)
    profile = dops.profile_from_report(report, 0.10, audit_rate=0.05)
    path = dops.save_profile(profile, tmp_path / "p.json")
    loaded = dops.load_profile(path)
    loaded.check_contract(contract)

    decisions = dops.decide_many(preds, loaded)
    assert {d.action for d in decisions} <= {"auto", "escalate"}
    assert any(d.audit for d in decisions)
    assert report.levels[0].certificate.loss_bound == 2.0
