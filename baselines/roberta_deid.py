"""obi/deid_roberta_i2b2 baseline — RoBERTa fine-tuned on i2b2 de-identification.

Install:
    pip install transformers torch
"""

from __future__ import annotations

from .base import BaselineModel

# i2b2 label set → our schema keys
_I2B2_TO_SCHEMA: dict[str, str] = {
    "NAME":      "name",
    "AGE":       "age",
    "DATE":      "date",
    "LOCATION":  "place",
    "CONTACT":   "phone",   # i2b2 CONTACT covers phone/email/fax
    "ID":        "id",
    "PROFESSION": None,     # not in our schema — skip
    "HOSPITAL":  "place",
}


class RobertaDeidBaseline(BaselineModel):
    """Wraps obi/deid_roberta_i2b2 token-classification pipeline."""

    name = "RoBERTa-i2b2"

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
            # r["entity_group"] is e.g. "B-NAME" or after aggregation just "NAME"
            label = r["entity_group"].lstrip("BI-").strip()
            schema_key = _I2B2_TO_SCHEMA.get(label)
            if schema_key is None:
                continue
            text = r["word"].strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
