"""scispaCy medical NER baseline.

Install:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
"""

from __future__ import annotations

from .base import BaselineModel

# spaCy (OntoNotes) entity types → our schema keys.
# Only the standard typed-NER labels map to PHI; everything else is skipped.
# Location split (verified on the test set): spaCy tags street addresses as FAC
# ("246 Maple Lane") and cities/states as GPE ("San Francisco", "WA"), which are
# our `address`; institutions are ORG ("Cleveland Clinic"), which is our `place`.
# NOTE: en_core_sci_lg only emits a single untyped "ENTITY" label, so none of
# these match and it yields no typed predictions (supported_entities below ends
# up empty) — use a typed model such as en_core_web_lg.
_SCI_TO_SCHEMA: dict[str, str] = {
    "PERSON":  "name",
    "ORG":     "place",     # organizations / institutions (hospitals, clinics)
    "FAC":     "address",   # facilities incl. street addresses
    "GPE":     "address",   # geopolitical: cities, states, countries
    "LOC":     "address",   # non-GPE locations: streets, parks
    "DATE":    "date",
    "TIME":    "date",
}


class ScispaCyBaseline(BaselineModel):
    """Wraps scispaCy (en_core_sci_lg) for clinical NER.

    Note: scispaCy is not specifically trained for PHI de-identification.
    It serves as a medical-domain NLP baseline to show the gap between
    general clinical NER and targeted de-id.
    """

    name = "scispaCy"

    def __init__(self, model_name: str = "en_core_web_lg"):
        import spacy
        self._nlp = spacy.load(model_name)
        # Scope macro-F1 / per-entity to the schema keys this model can actually
        # reach, derived from its NER labels (e.g. en_core_web_lg has no
        # phone/email/id/url/age/device labels -> those show as N/A, not 0.0).
        ner = self._nlp.get_pipe("ner") if "ner" in self._nlp.pipe_names else None
        labels = set(ner.labels) if ner is not None else set()
        self.supported_entities = {
            _SCI_TO_SCHEMA[lbl] for lbl in labels if _SCI_TO_SCHEMA.get(lbl)
        }

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        doc = self._nlp(note)
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            schema_key = _SCI_TO_SCHEMA.get(ent.label_)
            if schema_key is None:
                continue
            text = ent.text.strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
