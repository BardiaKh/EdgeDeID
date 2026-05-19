"""scispaCy medical NER baseline.

Install:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
"""

from __future__ import annotations

from .base import BaselineModel

# scispaCy entity types → our schema keys
# en_core_sci_lg uses BIONLP13CG or BC5CDR entities — mostly clinical, not de-id focused.
# We map the closest matches; many PHI types have no direct equivalent.
_SCI_TO_SCHEMA: dict[str, str] = {
    "PERSON":           "name",
    "GPE":              "place",   # geopolitical entity
    "LOC":              "place",
    "FAC":              "place",   # facility
    "ORG":              "place",
    "DATE":             "date",
    "TIME":             "date",
    "CARDINAL":         None,
    "DISEASE":          None,
    "CHEMICAL":         None,
    "GENE_OR_GENE_PRODUCT": None,
}


class ScispaCyBaseline(BaselineModel):
    """Wraps scispaCy (en_core_sci_lg) for clinical NER.

    Note: scispaCy is not specifically trained for PHI de-identification.
    It serves as a medical-domain NLP baseline to show the gap between
    general clinical NER and targeted de-id.
    """

    name = "scispaCy"

    def __init__(self, model_name: str = "en_core_sci_lg"):
        import spacy
        self._nlp = spacy.load(model_name)

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
