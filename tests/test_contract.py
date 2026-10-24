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


def test_contract_rejects_unknown_and_nonpositive_severity():
    with pytest.raises(ValidationError, match="severity for unknown choices"):
        DecisionContract(
            name="x", choices=["a", "b"], auto_choices=["a"], severity={"z": 2.0}
        )
    with pytest.raises(ValidationError, match="severity must be > 0"):
        DecisionContract(
            name="x", choices=["a", "b"], auto_choices=["a"], severity={"b": 0.0}
        )
    with pytest.raises(ValidationError, match="severity must be > 0"):
        DecisionContract(
            name="x", choices=["a", "b"], auto_choices=["a"], severity={"b": -1.0}
        )


def test_severity_defaults_and_loss_bound():
    assert ROUTING.miss_severity("deep") == 1.0
    assert ROUTING.loss_bound() == 1.0
    weighted = DecisionContract(
        name="triage",
        choices=["skip", "llm", "human"],
        auto_choices=["skip"],
        severity={"llm": 1.0, "human": 5.0},
    )
    assert weighted.miss_severity("skip") == 1.0
    assert weighted.miss_severity("llm") == 1.0
    assert weighted.miss_severity("human") == 5.0
    assert weighted.loss_bound() == 5.0
    assert weighted.summary()["severity"] == {"llm": 1.0, "human": 5.0}


def test_fingerprint_covers_severity_not_costs():
    base = ROUTING.fingerprint()
    # Costs are reporting-only: changing them must not invalidate a certificate.
    assert ROUTING.model_copy(update={"costs": {"skip": 0.0, "deep": 9.0}}).fingerprint() == base
    # Severity changes the certified loss, so the fingerprint must move.
    assert (
        ROUTING.model_copy(update={"severity": {"deep": 2.0}}).fingerprint() != base
    )


def test_escalate_choices_are_the_complement():
    assert ROUTING.escalate_choices == ["deep"]


def test_validate_data_rejects_predictions_outside_contract():
    case = DecisionCase(id="1", choices=["skip", "maybe"], label="skip")
    pred = DecisionPrediction(id="1", prediction="maybe", probabilities={"skip": 0.2, "maybe": 0.8})
    with pytest.raises(ValueError, match="does not fit contract"):
        ROUTING.validate_data([case], [pred])


def test_descriptions_must_name_known_choices():
    with pytest.raises(ValidationError, match="descriptions for unknown choices"):
        DecisionContract(
            name="x", choices=["a", "b"], auto_choices=["a"], descriptions={"z": "nope"}
        )


def test_descriptions_and_instructions_appear_in_summary():
    contract = DecisionContract(
        name="x",
        choices=["a", "b"],
        auto_choices=["a"],
        descriptions={"a": "safe to skip"},
        instructions="Route this.",
    )
    summary = contract.summary()
    assert summary["descriptions"] == {"a": "safe to skip"}
    assert summary["instructions"] == "Route this."


def test_laya_question_carries_descriptions_without_torch():
    from decisionops.adapters.laya import laya_question

    case = DecisionCase(id="1", choices=["skip", "deep"], label="skip", question="Q?")
    q = laya_question(case, descriptions={"deep": "touches auth"})
    assert q == {"t": "choice", "ins": "Q?", "crit": {"skip": None, "deep": "touches auth"}}
    # No descriptions -> identical to the pre-contract behaviour.
    assert laya_question(case)["crit"] == {"skip": None, "deep": None}
    assert laya_question(case, contract_instructions="C.")["ins"] == "C."


def test_yes_choice_requires_binary_contract_and_known_choice():
    with pytest.raises(ValidationError, match="exactly 2 choices"):
        DecisionContract(
            name="x", choices=["a", "b", "c"], auto_choices=["a"], yes_choice="b"
        )
    with pytest.raises(ValidationError, match="not in choices"):
        DecisionContract(name="x", choices=["a", "b"], auto_choices=["a"], yes_choice="z")
    ok = DecisionContract(name="x", choices=["no", "yes"], auto_choices=["no"], yes_choice="yes")
    assert ok.summary()["yes_choice"] == "yes"


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




def test_severity_weights_crc_loss_and_raises_bound():
    """
    Two equal-confidence misses, different true labels.

    With severity human=5 / llm=1, the human miss contributes 5× the loss mass,
    so the CRC bound needs a higher threshold (or becomes infeasible sooner)
    than plain 0-1 loss on the same pairs. B becomes 5, so min_n grows with it.
    """
    from decisionops.conformal import min_calibration_samples

    triage = DecisionContract(
        name="triage",
        choices=["skip", "llm", "human"],
        auto_choices=["skip"],
        severity={"llm": 1.0, "human": 5.0},
    )
    assert triage.loss_bound() == 5.0
    assert min_calibration_samples(0.05, triage.loss_bound()) == 99  # 5/0.05 - 1

    def _triage_pair(cid: str, pred: str, label: str, conf: float):
        others = [c for c in triage.choices if c != pred]
        rem = (1.0 - conf) / len(others)
        return (
            DecisionCase(id=cid, choices=list(triage.choices), label=label),
            DecisionPrediction(
                id=cid,
                prediction=pred,
                probabilities={pred: conf, **{o: rem for o in others}},
            ),
        )

    # 100 correct skips at conf 0.95, plus one llm miss and one human miss at 0.90.
    pairs = [_triage_pair(f"ok{i}", "skip", "skip", 0.95) for i in range(100)]
    pairs += [
        _triage_pair("miss_llm", "skip", "llm", 0.90),
        _triage_pair("miss_human", "skip", "human", 0.90),
    ]
    plain = certify_threshold(pairs, alpha=0.05, auto_choices=triage.auto_set)
    weighted = certify_threshold(
        pairs,
        alpha=0.05,
        auto_choices=triage.auto_set,
        severity_of=triage.miss_severity,
        loss_bound=triage.loss_bound(),
    )
    assert plain.feasible and plain.threshold is not None
    # At lambda=0.90 both misses are accepted: plain Rhat=2/102, weighted Rhat=6/102.
    # With B=5 the weighted bound is much larger, so the rule must raise the bar
    # past the miss confidences (or become infeasible).
    assert weighted.loss_bound == 5.0
    if weighted.feasible:
        assert weighted.threshold is not None
        assert weighted.threshold > plain.threshold
        assert weighted.threshold > 0.90
    # High-probability mode refuses non-0-1 severity weights.
    with pytest.raises(ValueError, match="0-1 loss"):
        certify_threshold(
            pairs,
            alpha=0.05,
            auto_choices=triage.auto_set,
            severity_of=triage.miss_severity,
            loss_bound=triage.loss_bound(),
            confidence_level=0.95,
        )


def test_run_certification_picks_up_contract_severity():
    triage = DecisionContract(
        name="triage",
        choices=["skip", "llm", "human"],
        auto_choices=["skip"],
        severity={"human": 4.0},
    )
    cases, preds = [], []
    for i in range(200):
        conf = 0.7 + (i % 30) / 100.0
        label = "skip" if i % 5 else ("human" if i % 2 == 0 else "llm")
        pred = "skip" if conf >= 0.75 else "llm"
        others = [c for c in triage.choices if c != pred]
        rem = (1.0 - conf) / len(others)
        cases.append(DecisionCase(id=str(i), choices=list(triage.choices), label=label))
        preds.append(
            DecisionPrediction(
                id=str(i),
                prediction=pred,
                probabilities={pred: conf, **{o: rem for o in others}},
            )
        )
    report = run_certification(
        cases, preds, alphas=[0.1], contract=triage, calibrate=False, seed=1
    )
    assert report.metadata["loss_bound"] == 4.0
    assert report.metadata["contract"]["severity"] == {"human": 4.0}
    assert any("severity" in n.lower() or "B=" in n for n in report.notes)
    level = report.levels[0]
    assert level.certificate.loss_bound == 4.0
    assert level.certificate.controlled_quantity == "weighted_joint_automated_error"


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
