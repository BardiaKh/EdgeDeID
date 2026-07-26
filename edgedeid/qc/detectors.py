"""High-recall deterministic PHI detectors (Tier-2 QC).

These cover the PHI types where pattern matching beats a small LM on *recall*:
email, phone, url, id, age, date. Contextual types (name / place / address)
are intentionally NOT detected here — they remain the model's responsibility
and are handled by the escalation path when the model is unreliable.

Each detector returns the verbatim span as it appears in the source note, so
reconciliation can substring-match against the model's tagged entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# entity types this module can detect (subset of the full PHI schema)
DETECTABLE_TYPES = ("email", "phone", "url", "id", "age", "date")


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected PHI span."""

    type: str       # one of DETECTABLE_TYPES
    text: str       # verbatim span from the source note
    start: int
    end: int
    subtype: str | None = None  # e.g. id subtype: MRN / SSN / NPI


# ─── individual patterns ──────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

_URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"')]+",
    re.IGNORECASE,
)

# (415) 555-2468 | 415-555-2468 | 415.555.2468 | +1 415 555 2468 | 4155552468
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d{1,2}[ .-]?)?"
    r"(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}"
    r"(?!\d)"
)

_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

# labelled identifiers: "MRN: AB-12345", "Medical Record Number is SH-1029384756",
# "NPI 1234567890", "account number 99887766"
_ID_LABELS = (
    r"(?:MRN|SSN|NPI|medical\s+record\s+(?:number|no\.?|#)|"
    r"record\s+number|health\s+plan(?:\s+id)?|account(?:\s+(?:number|no\.?|#))?|"
    r"billing\s+number|license(?:\s+(?:number|no\.?|#))?|patient\s+id)"
)
_LABELLED_ID_RE = re.compile(
    rf"{_ID_LABELS}\s*(?:is|:|#|=)?\s*([A-Z0-9][A-Z0-9.\-/]{{3,}})",
    re.IGNORECASE,
)

# age: "35y", "35 yo", "35 y/o", "35-year-old", "35 years old"  (number + unit)
#  OR  "age 35", "aged 35", "age: 35"                            (label + number)
_AGE_RE = re.compile(
    r"\b(?:"
    r"(\d{1,3})\s*(?:y(?:[/. ]?o)?\b|yo\b|years?[\s-]*old\b|[- ]year[- ]old\b)"
    r"|age[d]?\s*[:]?\s*(\d{1,3})\b"
    r")",
    re.IGNORECASE,
)

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
_DATE_RES = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                       # 2024-03-22
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                 # 03/22/2023
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),                 # 03-22-2023
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),  # March 22, 2023
    re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\.?\s+\d{{4}}\b", re.I),  # 22 March 2024
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{4}}\b", re.I),           # March 2024
)


def _id_subtype(label_match: str) -> str | None:
    lm = label_match.lower()
    if "ssn" in lm:
        return "SSN"
    if "npi" in lm:
        return "NPI"
    if "mrn" in lm or "medical record" in lm or "record number" in lm:
        return "MRN"
    if "health plan" in lm:
        return "health_plan"
    if "account" in lm:
        return "account"
    if "billing" in lm:
        return "billing_number"
    if "license" in lm:
        return "license"
    return "other"


def _non_overlapping(dets: list[Detection]) -> list[Detection]:
    """Drop spans fully contained in an already-kept (earlier-priority) span.

    Priority order is the insertion order of the callers below (email/url before
    phone/id so an email host isn't re-detected as something else).
    """
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: (x.start, -(x.end - x.start))):
        if any(d.start >= k.start and d.end <= k.end for k in kept):
            continue
        kept.append(d)
    return kept


# ─── public API ───────────────────────────────────────────────────────────────

def detect(note: str) -> list[Detection]:
    """Return all high-recall PHI detections in ``note`` (non-overlapping)."""
    if not isinstance(note, str) or not note:
        return []

    dets: list[Detection] = []

    for m in _EMAIL_RE.finditer(note):
        dets.append(Detection("email", m.group(0), m.start(), m.end()))
    for m in _URL_RE.finditer(note):
        dets.append(Detection("url", m.group(0).rstrip(".,);"), m.start(), m.end()))
    for m in _SSN_RE.finditer(note):
        dets.append(Detection("id", m.group(0), m.start(), m.end(), subtype="SSN"))
    for m in _LABELLED_ID_RE.finditer(note):
        val = m.group(1).rstrip(".,;:)")
        dets.append(
            Detection("id", val, m.start(1), m.start(1) + len(val),
                      subtype=_id_subtype(m.group(0)))
        )
    for m in _PHONE_RE.finditer(note):
        dets.append(Detection("phone", m.group(0).strip(), m.start(), m.end()))
    for m in _AGE_RE.finditer(note):
        gi = 1 if m.group(1) is not None else 2
        dets.append(Detection("age", m.group(gi), m.start(gi), m.end(gi)))
    for rx in _DATE_RES:
        for m in rx.finditer(note):
            dets.append(Detection("date", m.group(0), m.start(), m.end()))

    return _non_overlapping(dets)
