"""obi/deid_roberta_i2b2 baseline — RoBERTa fine-tuned on i2b2 de-identification.

Install:
    pip install transformers torch
"""

from __future__ import annotations

from .base import BaselineModel

# obi/deid_roberta_i2b2 emits BILOU-tagged labels over this base set:
#   AGE, DATE, EMAIL, HOSP, ID, LOC, OTHERPHI, PATIENT, PATORG, PHONE, STAFF
# Map each base label to our schema keys (name, age, date, place, address,
# phone, email, id, url, device). Labels with no schema equivalent map to None.
_I2B2_TO_SCHEMA: dict[str, str | None] = {
    "PATIENT":  "name",
    "STAFF":    "name",
    "AGE":      "age",
    "DATE":     "date",
    "HOSP":     "place",   # hospital/clinic institution names -> our `place`
    "LOC":      "address", # street/city/state/zip -> our `address` (GT `place`
                           # holds only institutions, which OBI tags as HOSP)
    "PHONE":    "phone",
    "EMAIL":    "email",
    "ID":       "id",
    "PATORG":   None,      # patient employer/organization — no schema key
    "OTHERPHI": None,      # catch-all bucket — no schema key
}


class RobertaDeidBaseline(BaselineModel):
    """Wraps obi/deid_roberta_i2b2 token-classification pipeline."""

    name = "RoBERTa-i2b2"

    # No URL or DEVICE label exists in the i2b2 2014 tag set, so this model
    # structurally cannot emit them — excluded from macro-F1 / per-entity.
    supported_entities = {
        "name", "age", "date", "place", "address", "phone", "email", "id",
    }

    def __init__(self, model_name: str = "obi/deid_roberta_i2b2"):
        from transformers import pipeline
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
        )

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        raw = self._pipe(note)
        entities: dict[str, list[str]] = {}
        for r in raw:
            # entity_group may carry a BILOU prefix (B-/I-/L-/U-); take the base
            # label after the final hyphen. ("U-PATIENT" -> "PATIENT")
            label = r["entity_group"].split("-")[-1].strip()
            schema_key = _I2B2_TO_SCHEMA.get(label)
            if schema_key is None:
                continue
            text = r["word"].strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
