"""OpenAI Privacy Filter (OPF) baseline — bidirectional token classifier w/ span decoding.

openai/privacy-filter is a 1.5B-param (50M active) PII detector with a 128k-token
context window and BIOES span decoding, covering 8 privacy categories (Apache 2.0).

It is NOT on PyPI — install from the cloned repo:
    git clone https://github.com/openai/privacy-filter
    pip install -e privacy-filter
Model weights auto-download from https://huggingface.co/openai/privacy-filter
to ~/.opf/privacy_filter on first use.
"""

from __future__ import annotations

from .base import BaselineModel

# OPF privacy categories -> our schema keys. "secret" (passwords/API keys/etc.)
# has no schema equivalent, so it maps to None and is skipped.
_OPF_TO_SCHEMA: dict[str, str | None] = {
    "private_person":  "name",
    "private_date":    "date",
    "private_address": "address",
    "private_phone":   "phone",
    "private_email":   "email",
    "private_url":     "url",
    "account_number":  "id",
    "secret":          None,
}


class OPFBaseline(BaselineModel):
    """Wraps the OpenAI Privacy Filter (opf._api.OPF) for PHI extraction."""

    name = "OPF"

    # OPF has no AGE, place/institution, or DEVICE category, so it structurally
    # cannot emit those — excluded from macro-F1 / per-entity reporting.
    supported_entities = {
        "name", "date", "address", "phone", "email", "id", "url",
    }

    def __init__(self, model=None, device: str | None = None):
        import torch
        from opf._api import OPF

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # output_mode="typed" keeps category-typed spans; detected_spans is
        # populated regardless, but typed mode is the documented default path.
        self._opf = OPF(model=model, device=device, output_mode="typed")

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        result = self._opf.redact(note)
        entities: dict[str, list[str]] = {}
        for span in result.detected_spans:
            schema_key = _OPF_TO_SCHEMA.get(span.label)
            if schema_key is None:
                continue
            text = span.text.strip()
            if text:
                entities.setdefault(schema_key, []).append(text)
        return entities
