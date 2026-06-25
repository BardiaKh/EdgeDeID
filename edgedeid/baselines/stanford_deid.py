"""StanfordAIMI/stanford-deidentifier-base baseline — BERT fine-tuned for PHI
de-identification (radiology + i2b2).

Install:
    pip install transformers torch
"""

from __future__ import annotations

from .base import BaselineModel

# stanford-deidentifier-base emits these typed labels (plus "O"):
#   VENDOR, DATE, HCW, HOSPITAL, ID, PATIENT, PHONE
# Map each to our schema keys. It has no label for age/address/email/url, so
# those schema keys are unreachable (excluded via supported_entities below).
_STANFORD_TO_SCHEMA: dict[str, str | None] = {
    "PATIENT":  "name",
    "HCW":      "name",     # healthcare worker (provider name)
    "DATE":     "date",
    "HOSPITAL": "place",
    "ID":       "id",
    "PHONE":    "phone",
    "VENDOR":   "device",   # equipment manufacturer; GT `device` strings lead
                            # with the vendor name (e.g. "Medtronic MiniMed..."),
                            # so this substring-matches device entities.
}


class StanfordDeidBaseline(BaselineModel):
    """Wraps StanfordAIMI/stanford-deidentifier-base token-classification."""

    name = "Stanford-deid"

    # No label exists for age/address/email/url — excluded from macro-F1 /
    # per-entity (shown as N/A rather than a misleading 0.0).
    supported_entities = {"name", "date", "place", "id", "phone", "device"}

    def __init__(self, model_name: str = "StanfordAIMI/stanford-deidentifier-base"):
        from transformers import edgedeid
        # "first" assigns each WordPiece word the label of its first sub-token,
        # which avoids the fragmentation/over-prediction seen with "simple" on
        # subword models. (Aggregation strategy can be revisited in lab.)
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="first",
        )

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        raw = self._pipe(note)
        entities: dict[str, list[str]] = {}
        for r in raw:
            # labels are plain (e.g. "DATE"); split guards against any B-/I- prefix
            label = r["entity_group"].split("-")[-1].strip()
            schema_key = _STANFORD_TO_SCHEMA.get(label)
            if schema_key is None:
                continue
            text = r["word"].strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
