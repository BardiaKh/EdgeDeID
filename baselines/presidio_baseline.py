"""Microsoft Presidio baseline for PHI de-identification.

Install:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

from .base import BaselineModel

# Mapping from Presidio entity types to our schema keys
_PRESIDIO_TO_SCHEMA: dict[str, str] = {
    "PERSON":          "name",
    "AGE":             "age",
    "DATE_TIME":       "date",
    "LOCATION":        "place",
    "ADDRESS":         "address",
    "PHONE_NUMBER":    "phone",
    "EMAIL_ADDRESS":   "email",
    "MEDICAL_LICENSE": "id",
    "NRP":             "id",
    "ID":              "id",
    "URL":             "url",
    "MEDICAL_DEVICE":  "device",
    "US_SSN":          "id",
    "US_DRIVER_LICENSE": "id",
    "US_PASSPORT":     "id",
    "US_ITIN":         "id",
    "US_BANK_NUMBER":  "id",
    "CREDIT_CARD":     "id",
    "IBAN_CODE":       "id",
    "IP_ADDRESS":      "id",
}


class PresidioBaseline(BaselineModel):
    """Wraps Microsoft Presidio Analyzer for PHI entity extraction."""

    name = "Presidio"

    def __init__(self, score_threshold: float = 0.5):
        from presidio_analyzer import AnalyzerEngine
        self._analyzer = AnalyzerEngine()
        self._threshold = score_threshold

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        results = self._analyzer.analyze(
            text=note,
            language="en",
            score_threshold=self._threshold,
        )
        entities: dict[str, list[str]] = {}
        for r in results:
            schema_key = _PRESIDIO_TO_SCHEMA.get(r.entity_type)
            if schema_key is None:
                continue
            span_text = note[r.start:r.end].strip()
            if span_text:
                entities.setdefault(schema_key, []).append(span_text)
        return entities
