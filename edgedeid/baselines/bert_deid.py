"""obi/deid_bert_i2b2 baseline — ClinicalBERT fine-tuned on i2b2 de-identification.

The ClinicalBERT sibling of obi/deid_roberta_i2b2 (same OBI team, same i2b2 2014
tag set), so it reuses the RoBERTa wrapper's label map and only swaps the model.
Lets us compare BERT vs RoBERTa vs the Qwen3 student on an identical label space.

Install:
    pip install transformers torch
"""

from __future__ import annotations

from .roberta_deid import RobertaDeidBaseline


class BertDeidBaseline(RobertaDeidBaseline):
    """Wraps obi/deid_bert_i2b2 — identical i2b2 tag set to the RoBERTa model."""

    name = "BERT-i2b2"

    def __init__(self, model_name: str = "obi/deid_bert_i2b2"):
        super().__init__(model_name=model_name)
