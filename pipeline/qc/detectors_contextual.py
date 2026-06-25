"""Tier-3 QC: contextual NER detector for name / place / address.

Regex (Tier-2) cannot reliably detect person names, place names, or
addresses — these require context. This module provides a lightweight
spaCy-based fallback for those three categories.

Cost contract
-------------
This module is expensive relative to regex, so it is ONLY triggered on
samples already flagged as escalated by Tier-1 or Tier-2.
It is NEVER run unconditionally on every note.

Dependencies
------------
    pip install spacy
    python -m spacy download en_core_web_sm

If spaCy or the model is unavailable, detect_contextual() raises
ImportError with a clear install instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTEXTUAL_TYPES = ("name", "place", "address")

# spaCy entity labels → PHI schema types
_LABEL_MAP: dict[str, str] = {
    "PERSON": "name",
    "GPE": "place",   # geopolitical entity: city, state, country
    "LOC": "place",   # non-GPE location (mountain, river, region)
    "FAC": "place",   # facility: hospital, clinic, building
    "ORG": "place",   # organisation — hospital names fall here
}

# Street-address heuristic: "123 Main Street", "456 Oak Ave", etc.
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z0-9\s]{2,35}"
    r"(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Lane|Ln\.?|"
    r"Drive|Dr\.?|Court|Ct\.?|Way|Place|Pl\.?|Circle|Cir\.?|"
    r"Highway|Hwy\.?|Terrace|Ter\.?|Trail|Trl\.?)\b",
    re.IGNORECASE,
)

# Lazy-loaded spaCy pipeline (one load per process)
_NLP = None


@dataclass(frozen=True, slots=True)
class ContextualDetection:
    type: str    # "name" | "place" | "address"
    text: str
    start: int
    end: int
    source: str  # "spacy" | "regex"


def _load_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "spaCy is required for Tier-3 QC.\n"
            "Install: pip install spacy && python -m spacy download en_core_web_sm"
        )
    try:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise ImportError(
            "spaCy model 'en_core_web_sm' not found.\n"
            "Install: python -m spacy download en_core_web_sm"
        )
    return _NLP


def detect_contextual(note: str) -> list[ContextualDetection]:
    """Return name / place / address detections for an escalated note.

    Combines spaCy NER (PERSON/GPE/LOC/FAC/ORG) with a street-address
    regex. Spans are deduplicated; regex spans already covered by a spaCy
    entity are dropped.
    """
    if not isinstance(note, str) or not note.strip():
        return []

    nlp = _load_nlp()
    doc = nlp(note)

    dets: list[ContextualDetection] = []
    covered: list[tuple[int, int]] = []

    for ent in doc.ents:
        phi_type = _LABEL_MAP.get(ent.label_)
        if phi_type is None or not ent.text.strip():
            continue
        dets.append(
            ContextualDetection(phi_type, ent.text.strip(),
                                ent.start_char, ent.end_char, "spacy")
        )
        covered.append((ent.start_char, ent.end_char))

    for m in _ADDRESS_RE.finditer(note):
        # skip spans already covered by a spaCy entity
        if any(s <= m.start() and m.end() <= e for s, e in covered):
            continue
        dets.append(
            ContextualDetection("address", m.group(0).strip(),
                                m.start(), m.end(), "regex")
        )
        covered.append((m.start(), m.end()))

    return dets
