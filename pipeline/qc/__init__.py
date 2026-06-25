"""Uncertainty-gated layered QC for PHI de-identification.

Tier-1: intrinsic metacognitive <qc_status> (trained via GRPO reward).
Tier-2: deterministic structural repair + high-recall regex detectors +
        reconcile recall floor  (email, phone, url, id, age, date).
Tier-3: contextual NER (spaCy en_core_web_sm) for name / place / address;
        triggered only on escalated samples to preserve edge latency.
"""

from .calibration import (
    QCCalibrationResult,
    calibration_to_dict,
    compute_qc_calibration,
    print_calibration_report,
)
from .detectors import DETECTABLE_TYPES, Detection, detect
from .detectors_contextual import CONTEXTUAL_TYPES, ContextualDetection, detect_contextual
from .pipeline import (
    EscalationFn,
    QCConfig,
    QCResult,
    apply_qc_batch,
    run_qc,
)
from .reconcile import ReconcileResult, reconcile
from .repair import RepairResult, empty_phi_xml, repair_xml

__all__ = [
    # Tier-2 detectors
    "Detection",
    "DETECTABLE_TYPES",
    "detect",
    # Tier-3 contextual detector
    "ContextualDetection",
    "CONTEXTUAL_TYPES",
    "detect_contextual",
    # repair & reconcile
    "RepairResult",
    "repair_xml",
    "empty_phi_xml",
    "ReconcileResult",
    "reconcile",
    # pipeline
    "QCConfig",
    "QCResult",
    "EscalationFn",
    "run_qc",
    "apply_qc_batch",
    # calibration
    "QCCalibrationResult",
    "compute_qc_calibration",
    "print_calibration_report",
    "calibration_to_dict",
]
