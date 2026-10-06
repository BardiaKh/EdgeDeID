"""Uncertainty-gated layered QC orchestrator (Tier-1 + Tier-2 + Tier-3).

Tier-1  — intrinsic metacognitive <qc_status> tag (GRPO-trained).
Tier-2  — deterministic structural repair + high-recall regex detectors
          (email, phone, url, id, age, date) + reconcile recall floor.
Tier-3  — contextual NER (spaCy) for name / place / address; only triggered
          on escalated samples to keep on-device latency low.

Flow per sample:

    model output ─┐
                  ├─► structural repair ──► reconcile(T2 detectors) ──► T2 XML
    source note ──┘            │                     │
                               ▼                     ▼
                       (unrecoverable?)        (n_injected_t2)
                               │                     │
   intrinsic <qc_status> ──────┴─────► escalation gate
                                              │
                                    escalated? (any reason)
                                              │
                              ┌───────────────┼─────────────────┐
                              │ use_tier3     │                  │
                              ▼               ▼                  ▼
                     Tier-3 NER         escalation_fn        (no extra step)
                     reconcile(T3)      (external model)
                              │
                              ▼
                         final XML
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
    escalate_on_missing_qc: bool = True
    inject_escalate_threshold: int = 3   # many T2 injections → model is unreliable
    use_tier3: bool = False              # enable Tier-3 spaCy NER on escalated samples
    tier3_on_escalated_only: bool = True  # if False, run T3 on every sample (expensive)
    escalation_fn: EscalationFn | None = None  # opt-in expensive handler
    # v2 precision fixes (2026-10-05); all off by default so v1 stays reproducible.
    lenient_repair: bool = False     # recovering-parser fallback for mismatched tags
    strict_dates: bool = False       # Tier-2: drop numeric "dates" with impossible month/day
    precision_guards: bool = False   # Tier-3: abbreviation/institution/case guards


@dataclass(slots=True)
class QCResult:
    final_xml: str
    escalated: bool
    reasons: list[str] = field(default_factory=list)
    intrinsic_qc: str | None = None    # "complete" | "incomplete" | None
    repair_status: str = "ok"          # ok | repaired | unrecoverable
    n_injected: int = 0                # Tier-2 injections
    n_injected_tier3: int = 0          # Tier-3 injections
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

    # ── Tier-2: structural repair + regex detectors + reconcile ───────────────
    t2_detections = detect(source_note, strict_dates=cfg.strict_dates)
    rep = repair_xml(model_output, lenient=cfg.lenient_repair)
    if rep.root is None:
        root = ET.fromstring(empty_phi_xml())
        repair_status = "unrecoverable"
    else:
        root = rep.root
        repair_status = rep.status

    rec_t2 = reconcile(root, t2_detections)

    # ── escalation decision ───────────────────────────────────────────────────
    reasons: list[str] = []
    if repair_status == "unrecoverable":
        reasons.append("unrecoverable_xml")
    if intrinsic == "incomplete":
        reasons.append("model_self_flag_incomplete")
    if intrinsic is None and cfg.escalate_on_missing_qc:
        reasons.append("missing_qc_tag")
    if rec_t2.n_injected >= cfg.inject_escalate_threshold:
        reasons.append(f"high_t2_injections({rec_t2.n_injected})")

    escalated = bool(reasons)
    final_xml = rec_t2.xml
    n_injected_tier3 = 0

    audit: dict = {
        "intrinsic_qc": intrinsic,
        "repair_status": repair_status,
        "n_detected_t2": len(t2_detections),
        "n_injected_t2": rec_t2.n_injected,
        "injected_t2": [(d.type, d.text) for d in rec_t2.injected],
        "reasons": reasons,
        "escalated": escalated,
    }

    # ── Tier-3: contextual NER for name / place / address ────────────────────
    run_t3 = cfg.use_tier3 and (escalated or not cfg.tier3_on_escalated_only)
    if run_t3:
        from .detectors_contextual import detect_contextual  # lazy import (heavy)
        t3_detections = detect_contextual(source_note, guarded=cfg.precision_guards)
        if t3_detections:
            # re-parse to get a fresh mutable root from the T2-reconciled XML
            try:
                root_t3 = ET.fromstring(final_xml)
            except ET.ParseError:
                root_t3 = ET.fromstring(empty_phi_xml())
            rec_t3 = reconcile(root_t3, t3_detections)  # type: ignore[arg-type]
            final_xml = rec_t3.xml
            n_injected_tier3 = rec_t3.n_injected
            audit["n_detected_t3"] = len(t3_detections)
            audit["n_injected_t3"] = rec_t3.n_injected
            audit["injected_t3"] = [(d.type, d.text) for d in rec_t3.injected]
        else:
            audit["n_detected_t3"] = 0
            audit["n_injected_t3"] = 0
            audit["injected_t3"] = []

    # ── optional external escalation handler ─────────────────────────────────
    if escalated and cfg.escalation_fn is not None:
        replacement = cfg.escalation_fn(source_note, final_xml, dict(audit))
        rep2 = repair_xml(replacement, lenient=cfg.lenient_repair)
        if rep2.root is not None:
            rec2 = reconcile(rep2.root, t2_detections)
            final_xml = rec2.xml
            audit["escalation_applied"] = True
            audit["post_escalation_injected_t2"] = rec2.n_injected
        else:
            audit["escalation_applied"] = False

    return QCResult(
        final_xml=final_xml,
        escalated=escalated,
        reasons=reasons,
        intrinsic_qc=intrinsic,
        repair_status=repair_status,
        n_injected=rec_t2.n_injected,
        n_injected_tier3=n_injected_tier3,
        audit=audit,
    )


def apply_qc_batch(
    predictions: list[str],
    source_notes: list[str],
    config: QCConfig | None = None,
    qc_statuses: list[str | None] | None = None,
) -> tuple[list[str], list[QCResult]]:
    """Vectorised QC over a list of (prediction, note) pairs.

    Returns (final_xml_list, qc_result_list). Feed final_xml_list directly
    into pipeline.eval.compute_metrics to score a given QC configuration.
    Use qc_statuses to override the intrinsic <qc_status> tag (e.g. for
    ablation where you want to force-escalate everything).
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
