"""Reconcile model entities with deterministic detections (Tier-2 QC).

Guarantees a *recall floor*: every pattern-detectable PHI span the model
failed to tag is injected into the schema, so downstream recall for
email/phone/url/id/age/date cannot fall below the detector's recall —
regardless of how weak the policy is.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime

from .detectors import Detection
from .repair import SCHEMA

_TYPE_TO_PARENT = {child: parent for parent, child in SCHEMA}
_MIN_MATCH_LEN = 3  # consistent with pipeline.rl.rewards._soft_match


def _soft_covered(span: str, existing: list[str]) -> bool:
    """True if the model already tagged something matching ``span``."""
    s = span.strip().lower()
    if not s:
        return True
    for e in existing:
        e = e.strip().lower()
        if not e:
            continue
        if s == e:
            return True
        if min(len(s), len(e)) >= _MIN_MATCH_LEN and (s in e or e in s):
            return True
    return False


def _all_existing_texts(root: ET.Element) -> list[str]:
    """Every leaf text already present in the tree (any entity type)."""
    out: list[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            out.append(elem.text.strip())
    return out


def _ensure_parent(root: ET.Element, parent_tag: str) -> ET.Element:
    p = root.find(parent_tag)
    if p is None:
        p = ET.SubElement(root, parent_tag)
    return p


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y",
                 "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                 "%B %Y", "%b %Y", "%Y")


def _format_date(raw: str) -> str:
    r = raw.strip().replace(".", "")
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(r, fmt)
        except ValueError:
            continue
        if fmt in ("%B %Y", "%b %Y"):
            return f"{dt.year:04d}-{dt.month:02d}-??"
        if fmt == "%Y":
            return f"{dt.year:04d}-??-??"
        return dt.strftime("%Y-%m-%d")
    return "????-??-??"


def _inject(parent: ET.Element, det: Detection) -> None:
    if det.type == "date":
        d = ET.SubElement(parent, "date")
        ET.SubElement(d, "original").text = det.text
        ET.SubElement(d, "formatted").text = _format_date(det.text)
    elif det.type == "id":
        e = ET.SubElement(parent, "id")
        if det.subtype:
            e.set("type", det.subtype)
        e.text = det.text
    else:
        ET.SubElement(parent, det.type).text = det.text


@dataclass(slots=True)
class ReconcileResult:
    xml: str                       # final reconciled <phi_entities> string
    injected: list[Detection] = field(default_factory=list)
    n_injected: int = 0


def reconcile(root: ET.Element, detections: list[Detection]) -> ReconcileResult:
    """Inject every uncovered detection into ``root`` and serialise it."""
    existing = _all_existing_texts(root)
    injected: list[Detection] = []

    for det in detections:
        if _soft_covered(det.text, existing):
            continue
        parent_tag = _TYPE_TO_PARENT[det.type]
        _inject(_ensure_parent(root, parent_tag), det)
        existing.append(det.text)  # dedupe within this pass too
        injected.append(det)

    xml = ET.tostring(root, encoding="unicode")
    # tidy self-closed empties for readability/consistency with the schema
    xml = re.sub(r"<(\w+)></\1>", r"<\1/>", xml)
    return ReconcileResult(xml=xml, injected=injected, n_injected=len(injected))
