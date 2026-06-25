"""OpenMed PII baseline — modern (2025) transformer fine-tuned for PII/PHI.

OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1 is a DeBERTa-v3-large token
classifier covering 54 PII entity types (Apache 2.0). Unlike the i2b2 models it
has dedicated labels for medical record numbers, street addresses, and URLs, so
it can reach schema keys (id, address, url) where Presidio/scispaCy fail.

Install:
    pip install transformers torch sentencepiece
"""

from __future__ import annotations

from .base import BaselineModel

# OpenMed base labels (BIO prefix stripped) -> our schema keys. The model emits
# 54 PII types; only those that align with our PHI schema are mapped, everything
# else (passwords, blood_type, gender, ip, ...) maps to None and is skipped.
# Notes on the judgement calls:
#   company_name -> place : no dedicated hospital label exists; institutions are
#                           the closest fit (GT `place` holds only institutions).
#                           May add employer false positives — revisit in lab.
#   fax_number   -> phone : HIPAA treats fax as a telephone-type identifier.
#   *_id / *_number -> id : generic unique identifiers (MRN, SSN, etc.).
_OPENMED_TO_SCHEMA: dict[str, str] = {
    "first_name":                     "name",
    "last_name":                      "name",
    "age":                            "age",
    "date":                           "date",
    "date_of_birth":                  "date",
    "date_time":                      "date",
    "time":                           "date",
    "street_address":                 "address",
    "city":                           "address",
    "county":                         "address",
    "state":                          "address",
    "country":                        "address",
    "postcode":                       "address",
    "coordinate":                     "address",
    "company_name":                   "place",
    "phone_number":                   "phone",
    "fax_number":                     "phone",
    "email":                          "email",
    "url":                            "url",
    "medical_record_number":          "id",
    "health_plan_beneficiary_number": "id",
    "account_number":                 "id",
    "ssn":                            "id",
    "tax_id":                         "id",
    "certificate_license_number":     "id",
    "employee_id":                    "id",
    "customer_id":                    "id",
    "unique_id":                      "id",
    "license_plate":                  "id",
    "vehicle_identifier":             "id",
    "device_identifier":              "device",
}


class OpenMedPIIBaseline(BaselineModel):
    """Wraps OpenMed-PII-SuperClinical token-classification for PHI extraction."""

    name = "OpenMed-PII"

    # Every schema key is reachable via the map above, so the default full set
    # applies. Stated explicitly for clarity / parity with the other baselines.
    supported_entities = {
        "name", "age", "date", "place", "address",
        "phone", "email", "id", "url", "device",
    }

    def __init__(
        self,
        model_name: str = "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1",
    ):
        from transformers import edgedeid
        # "first" assigns each word the label of its first sub-token — matches
        # the aggregation strategy used by the other transformer baselines.
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="first",
        )

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        raw = self._pipe(note)
        entities: dict[str, list[str]] = {}
        for r in raw:
            # entity_group is the base label (BIO prefix already merged); the
            # only hyphen would be a leftover B-/I- prefix, so split guards it.
            # Labels use underscores ("first_name"), which are preserved.
            label = r["entity_group"].split("-")[-1].strip().lower()
            schema_key = _OPENMED_TO_SCHEMA.get(label)
            if schema_key is None:
                continue
            text = r["word"].strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
