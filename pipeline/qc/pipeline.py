"""Uncertainty-gated layered QC orchestrator (Tier-1 + Tier-2).

Flow per sample:

    model output ─┐
                  ├─► structural repair ──► reconcile w/ detectors ──► final XML
    source note ──┘            │                     │
                               ▼                     ▼
                       (unrecoverable?)        (n_injected)
                               │                     │
   intrinsic <qc_status> ──────┴───────► escalation gate ──► [escalation_fn?]

The cheap intrinsic signal (GRPO-trained <qc_status>) plus deterministic
residuals decide whether a sample gets extra (expensive) compute. The default
path is fully deterministic and edge-friendly; escalation is opt-in.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable

from .detectors import detect
from .reconcile import reconcile
from .repair import empty_phi_xml, repair_xml

EscalationFn = Callable[[str, str, dict], str]
"""(source_note, current_xml, audit) -> replacement model output / XML."""

_QC_RE = re.compile(r"<qc_status\s*>(complete|incomplete)</qc_status>", re.IGNORECASE)


@dataclass(slots=True)
class QCConfig:
    escalate_on_missing_qc: bool = True   # no <qc_status> tag ⇒ treat as uncertain
    inject_escalate_threshold: int = 3    # many deterministic misses ⇒ model unreliable
    escalation_fn: EscalationFn | None = None  # None ⇒ conservative recall-floor only


@dataclass(slots=True)
class QCResult:
    final_xml: str
    escalated: bool
    reasons: list[str] = field(default_factory=list)
    intrinsic_qc: str | None = None       # "complete" | "incomplete" | None
    repair_status: str = "ok"             # ok | repaired | unrecoverable
    n_injected: int = 0
    audit: dict = field(default_factory=dict)


def _extract_qc(text: str) -> str | None:
    m = _QC_RE.search(text or "")
    return m.group(1).lower() if m else None


def run_qc(
    model_output: str,
    source_note: str,
    config: QCConfig | None = None,
    qc_status: str | None = None,
) -> QCResult:
    """Run the full layered QC pipeline on one (output, note) pair."""
    cfg = config or QCConfig()
    intrinsic = qc_status or _extract_qc(model_output)
    detections = detect(source_note)

    rep = repair_xml(model_output)
    if rep.root is None:
        root = ET.fromstring(empty_phi_xml())
        repair_status = "unrecoverable"
    else:
        root = rep.root
        repair_status = rep.status

    rec = reconcile(root, detections)

    reasons: list[str] = []
    if repair_status == "unrecoverable":
        reasons.append("unrecoverable_xml")
    if intrinsic == "incomplete":
        reasons.append("model_self_flag_incomplete")
    if intrinsic is None and cfg.escalate_on_missing_qc:
        reasons.append("missing_qc_tag")
    if rec.n_injected >= cfg.inject_escalate_threshold:
        reasons.append(f"high_residual_injections({rec.n_injected})")

    escalated = bool(reasons)
    final_xml = rec.xml

    audit = {
        "intrinsic_qc": intrinsic,
        "repair_status": repair_status,
        "n_detected": len(detections),
        "n_injected": rec.n_injected,
        "injected": [(d.type, d.text) for d in rec.injected],
        "reasons": reasons,
    }

    if escalated and cfg.escalation_fn is not None:
        replacement = cfg.escalation_fn(source_note, final_xml, dict(audit))
        rep2 = repair_xml(replacement)
        if rep2.root is not None:
            rec2 = reconcile(rep2.root, detections)  # keep the recall floor
            final_xml = rec2.xml
            audit["escalation_applied"] = True
            audit["post_escalation_injected"] = rec2.n_injected
        else:
            audit["escalation_applied"] = False  # handler output unusable; keep safe XML

    return QCResult(
        final_xml=final_xml,
        escalated=escalated,
        reasons=reasons,
        intrinsic_qc=intrinsic,
        repair_status=repair_status,
        n_injected=rec.n_injected,
        audit=audit,
    )


def apply_qc_batch(
    predictions: list[str],
    source_notes: list[str],
    config: QCConfig | None = None,
    qc_statuses: list[str | None] | None = None,
) -> tuple[list[str], list[QCResult]]:
    """Vectorised QC for evaluation / ablation (rows D and F).

    Returns (final_xml_list, qc_result_list); feed the first into
    ``pipeline.eval.compute_metrics`` to score the QC-augmented system.
    """
    cfg = config or QCConfig()
    if qc_statuses is None:
        qc_statuses = [None] * len(predictions)
    finals: list[str] = []
    results: list[QCResult] = []
    for pred, note, qs in zip(predictions, source_notes, qc_statuses):
        r = run_qc(pred, note, cfg, qc_status=qs)
        finals.append(r.final_xml)
        results.append(r)
    return finals, results
