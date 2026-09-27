"""Decision contract schema, validation, and asymmetric-risk behaviour."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from decisionops.conformal import certify_threshold
from decisionops.conformal.certify import run_certification
from decisionops.contract import DecisionContract, load_contract
from decisionops.eval import run_gate
from decisionops.schemas import DecisionCase, DecisionPrediction
from decisionops.selective.threshold import evaluate_threshold

ROUTING = DecisionContract(
    name="routing",
    choices=["skip", "deep"],
    auto_choices=["skip"],
    costs={"skip": 0.0, "deep": 1.0},
)


def _pair(cid: str, pred: str, label: str, conf: float):
    other = "deep" if pred == "skip" else "skip"
    return (
        DecisionCase(id=cid, choices=["skip", "deep"], label=label),
        DecisionPrediction(
            id=cid, prediction=pred, probabilities={pred: conf, other: 1.0 - conf}
        ),
    )


# ------------------------------------------------------------------ schema


def test_contract_rejects_auto_choice_outside_choices():
    with pytest.raises(ValidationError, match="auto_choices not in choices"):
        DecisionContract(name="x", choices=["a", "b"], auto_choices=["c"])


def test_contract_rejects_duplicates_unknown_and_negative_costs():
    with pytest.raises(ValidationError, match="duplicate"):
        DecisionContract(name="x", choices=["a", "a"], auto_choices=["a"])
    with pytest.raises(ValidationError, match="unknown choices"):
        DecisionContract(name="x", choices=["a", "b"], auto_choices=["a"], costs={"z": 1})
    with pytest.raises(ValidationError, match="negative"):
        DecisionContract(name="x", choices=["a", "b"], auto_choices=["a"], costs={"b": -1})


def test_escalate_choices_are_the_complement():
    assert ROUTING.escalate_choices == ["deep"]


def test_validate_data_rejects_predictions_outside_contract():
    case = DecisionCase(id="1", choices=["skip", "maybe"], label="skip")
    pred = DecisionPrediction(id="1", prediction="maybe", probabilities={"skip": 0.2, "maybe": 0.8})
    with pytest.raises(ValueError, match="does not fit contract"):
        ROUTING.validate_data([case], [pred])


def test_load_contract_round_trips(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(ROUTING.model_dump()), encoding="utf-8")
    assert load_contract(path) == ROUTING
    with pytest.raises(FileNotFoundError):
        load_contract(tmp_path / "missing.json")


# ------------------------------------------------------ asymmetric risk


def test_wrong_escalations_are_not_risk():
    # All "deep" predictions are wrong (the case did not need deep review).
    # That only costs money. With a contract they must not count as risk.
    pairs = [_pair(f"s{i}", "skip", "skip", 0.95) for i in range(8)]
    pairs += [_pair(f"d{i}", "deep", "skip", 0.95) for i in range(2)]

    symmetric = evaluate_threshold(pairs, 0.9)
    assert symmetric.selective_risk == pytest.approx(0.2)

    asym = evaluate_threshold(pairs, 0.9, auto_choices=ROUTING.auto_set)
    assert asym.n_accepted == 8  # escalations are never auto-executed
    assert asym.selective_risk == 0.0
    # Coverage stays a fraction of *all* traffic.
    assert asym.coverage == pytest.approx(0.8)


def test_missed_review_still_counts_as_risk():
    # "skip" predicted but the change needed deep review: that is the risk.
    pairs = [_pair(f"s{i}", "skip", "skip", 0.95) for i in range(9)]
    pairs.append(_pair("miss", "skip", "deep", 0.95))
    point = evaluate_threshold(pairs, 0.9, auto_choices=ROUTING.auto_set)
    assert point.selective_risk == pytest.approx(0.1)


def test_crc_ignores_escalations_and_certifies_more_coverage():
    # 200 correct skips + 60 wrong escalations, all at high confidence.
    # Symmetric CRC sees 60 errors and cannot certify 5%; asymmetric sees none.
    pairs = [_pair(f"s{i}", "skip", "skip", 0.97) for i in range(200)]
    pairs += [_pair(f"d{i}", "deep", "skip", 0.97) for i in range(60)]

    symmetric = certify_threshold(pairs, alpha=0.05)
    asym = certify_threshold(pairs, alpha=0.05, auto_choices=ROUTING.auto_set)

    assert not symmetric.feasible
    assert asym.feasible
    assert asym.empirical_joint_error == 0.0
    assert asym.coverage_calibration == pytest.approx(200 / 260)


def test_crc_infeasible_when_nothing_is_on_the_auto_path():
    pairs = [_pair(f"d{i}", "deep", "deep", 0.9) for i in range(150)]
    result = certify_threshold(pairs, alpha=0.05, auto_choices=ROUTING.auto_set)
    assert not result.feasible
    assert any("automatic path" in n for n in result.notes)


def test_asymmetric_crc_stays_conservative_against_closed_form():
    """
    Same closed-form check as test_crc_analytic, but with half the traffic
    escalated and every escalation wrong.

    A correct asymmetric rule must ignore the escalations entirely. On the auto
    half, confidence ~ U(0.55, 0.999) is perfectly calibrated, so with auto share
    q = 0.5 the true joint miss rate at lambda is
        q * ((1-lam)^2 - (1-HI)^2) / (2 * SPAN)
    and lambda_hat must sit at or above the lambda* that makes it equal alpha.
    If escalation errors leaked into the loss, lambda_hat would be pushed far
    above lambda* (or be infeasible), and the implied risk would be far below
    alpha — so the test also bounds the gap from above.
    """
    lo, hi = 0.55, 0.999
    span = hi - lo
    q = 0.5
    alpha = 0.05
    lam_star = 1.0 - math.sqrt(2.0 * (span * alpha / q + (1 - hi) ** 2 / 2.0))

    lams = []
    for rep in range(60):
        rng = np.random.default_rng(70_000 + rep)
        pairs = []
        for i in range(300):
            if rng.random() < q:
                conf = float(rng.uniform(lo, hi))
                label = "skip" if rng.random() < conf else "deep"
                pairs.append(_pair(str(i), "skip", label, conf))
            else:
                # Escalated and wrong: must contribute zero loss.
                pairs.append(_pair(str(i), "deep", "skip", float(rng.uniform(lo, hi))))
        res = certify_threshold(pairs, alpha, auto_choices=ROUTING.auto_set)
        if res.feasible:
            lams.append(res.threshold)

    assert len(lams) >= 48
    arr = np.asarray(lams)
    gap = float(arr.mean() - lam_star)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    # Conservative: not meaningfully below lambda*.
    assert gap > -2.0 * se, f"gap {gap:+.5f} se {se:.5f}: anti-conservative"
    # Not leaking escalation errors: close to lambda*, not pinned near 1.
    assert gap < 0.05, f"gap {gap:+.5f}: escalations appear to count as loss"


# --------------------------------------------------------- end to end


def test_gate_and_certify_record_contract(tmp_path):
    rng = np.random.default_rng(0)
    cases, preds = [], []
    for i in range(400):
        if i % 3 == 0:
            c, p = _pair(str(i), "deep", "skip", 0.9)
        else:
            conf = float(rng.uniform(0.6, 0.99))
            c, p = _pair(str(i), "skip", "skip" if rng.random() < conf else "deep", conf)
        cases.append(c)
        preds.append(p)

    gate = run_gate(cases, preds, calibrate=False, contract=ROUTING)
    assert gate.metadata["contract"]["auto_choices"] == ["skip"]
    for cond in gate.conditions:
        for t in cond.thresholds:
            # Escalations never auto-execute, so coverage cannot exceed the skip share.
            assert t.coverage <= 2 / 3 + 0.05

    cert = run_certification(cases, preds, calibrate=False, contract=ROUTING, alphas=[0.1])
    assert cert.metadata["contract"]["name"] == "routing"
    assert any("automated miss rate" in n for n in cert.notes)
