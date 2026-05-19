"""Uncertainty-gated layered QC for PHI de-identification.

Tier-1: intrinsic metacognitive <qc_status> (trained via GRPO reward).
Tier-2: deterministic structural repair + high-recall detectors + reconcile,
escalation gated by the Tier-1 signal.
"""

from .detectors import DETECTABLE_TYPES, Detection, detect
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
    "Detection",
    "DETECTABLE_TYPES",
    "detect",
    "RepairResult",
    "repair_xml",
    "empty_phi_xml",
    "ReconcileResult",
    "reconcile",
    "QCConfig",
    "QCResult",
    "EscalationFn",
    "run_qc",
    "apply_qc_batch",
]
