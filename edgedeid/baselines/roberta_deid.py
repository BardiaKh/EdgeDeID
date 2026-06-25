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
        from transformers import edgedeid
        # Aggregation strategy controls how byte-level subword tokens are merged
        # back into entity spans. "simple" leaves fragments ("Seattle" ->
        # "Se"+"attle") that inflate prediction counts ~6x and crater precision;
        # "average"/"max" fix that but suppress EMAIL entirely (the @/. word
        # boundaries flip the averaged label) and halve AGE. "first" gives the
        # best macro-F1 while preserving every entity.
        #
        # Per-entity F1 on data/test.csv (revise in lab):
        #   entity    simple   first    average
        #   name      0.572    0.687    0.737
        #   age       0.987    0.988    0.692
        #   date      0.595    0.454    0.506
        #   place     0.286    0.469    0.571
        #   address   0.309    0.405    0.490
        #   phone     0.649    0.652    0.654
        #   email     0.678    0.695    0.000
        #   id        0.422    0.640    0.741
        #   MACRO     0.562    0.624    0.549
        # _AGG = "simple"
        # _AGG = "average"
        _AGG = "first"
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy=_AGG,
        )
        # obi's tokenizers ship without model_max_length set (defaults to the
        # 1e30 sentinel), so the pipeline never truncates and crashes on notes
        # over 512 tokens. Both i2b2 models (BERT and RoBERTa) have a 512-token
        # effective limit, so pin it explicitly to make `stride` chunking in
        # predict_entities actually kick in.
        self._pipe.tokenizer.model_max_length = 512

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        # Notes can exceed the 512-token limit of the i2b2 BERT/RoBERTa models.
        # `stride` makes the pipeline split long input into overlapping windows
        # (512 tokens, 128-token overlap) and merge the results, instead of
        # crashing on the position-embedding add. Plain truncation would drop
        # every entity past token 512 and tank recall on long notes.
        raw = self._pipe(note, stride=128)
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
