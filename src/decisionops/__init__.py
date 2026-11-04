"""DecisionOps: reliability infrastructure for AI decision models."""

from decisionops.__version__ import __version__
from decisionops.cascade import CascadeReport, run_cascade
from decisionops.conformal import (
    CertificationReport,
    CRCResult,
    certify_cascade,
    certify_selective_threshold,
    certify_threshold,
    run_cascade_certification,
    run_certification,
)
from decisionops.contract import DecisionContract, load_contract
from decisionops.decide import (
    CascadeProfile,
    CertificationProfile,
    Decision,
    ProfileMismatchError,
    cascade_profile_from,
    cascade_profile_from_report,
    decide,
    decide_cascade,
    decide_cascade_many,
    decide_many,
    load_profile,
    profile_from_report,
    save_profile,
)
from decisionops.drift import DriftReport, run_drift_monitor
from decisionops.eval import evaluate, run_gate
from decisionops.io import (
    align_cases_and_predictions,
    load_dataset,
    load_decisions,
    load_outcomes,
    load_predictions,
)
from decisionops.recertify import RecertifyReport, run_recertification
from decisionops.schemas import (
    DecisionCase,
    DecisionPrediction,
    EvalResult,
    GateReport,
    OutcomeFeedback,
)

__all__ = [
    "CRCResult",
    "CascadeProfile",
    "CascadeReport",
    "CertificationProfile",
    "CertificationReport",
    "Decision",
    "DecisionCase",
    "DecisionContract",
    "DecisionPrediction",
    "EvalResult",
    "GateReport",
    "OutcomeFeedback",
    "ProfileMismatchError",
    "DriftReport",
    "RecertifyReport",
    "__version__",
    "align_cases_and_predictions",
    "cascade_profile_from",
    "cascade_profile_from_report",
    "certify_cascade",
    "certify_selective_threshold",
    "certify_threshold",
    "decide",
    "decide_cascade",
    "decide_cascade_many",
    "decide_many",
    "evaluate",
    "load_contract",
    "load_dataset",
    "load_decisions",
    "load_outcomes",
    "load_predictions",
    "load_profile",
    "profile_from_report",
    "run_cascade",
    "run_cascade_certification",
    "run_drift_monitor",
    "run_certification",
    "run_gate",
    "run_recertification",
    "save_profile",
]
