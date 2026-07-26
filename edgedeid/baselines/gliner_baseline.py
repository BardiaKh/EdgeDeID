"""GLiNER baseline — zero-shot generalist NER.

Install:
    pip install gliner
"""

from __future__ import annotations

from .base import BaselineModel

# Free-text labels we ask GLiNER to detect, mapped to our schema keys.
# This dict is the single source of truth: the prompt label list is derived
# from its keys, so the two can never drift out of sync.
_LABEL_TO_SCHEMA: dict[str, str] = {
    "person name":                          "name",
    "patient age":                          "age",
    "date":                                 "date",
    "hospital or clinic":                   "place",
    "address":                              "address",
    "phone number":                         "phone",
    "email address":                        "email",
    "medical record number or identifier":  "id",
    "url":                                  "url",
    "medical device":                       "device",
}

_GLINER_LABELS: list[str] = list(_LABEL_TO_SCHEMA)


class GLiNERBaseline(BaselineModel):
    """Wraps GLiNER for zero-shot PHI entity extraction."""

    name = "GLiNER"

    def __init__(
        self,
        model_name: str = "urchade/gliner_mediumv2.1",
        threshold: float = 0.5,
    ):
        from gliner import GLiNER
        self._model = GLiNER.from_pretrained(model_name)
        self._threshold = threshold

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        results = self._model.predict_entities(
            note, _GLINER_LABELS, threshold=self._threshold
        )
        entities: dict[str, list[str]] = {}
        for r in results:
            schema_key = _LABEL_TO_SCHEMA.get(r["label"])
            if schema_key is None:
                continue
            text = r["text"].strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
