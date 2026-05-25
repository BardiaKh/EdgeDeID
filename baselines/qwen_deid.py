"""Causal-LM PHI de-identification baseline (Qwen3-0.6B family).

Wraps a Qwen3-0.6B causal LM that emits <phi_entities> XML directly via the SFT
prompt template. Parameterized by checkpoint + display name so one class covers:
  * our fine-tuned student : EdgeDeID/Qwen3-0.6B-Base_SFT
  * the un-finetuned base   : Qwen/Qwen3-0.6B-Base  (zero-shot reference)

Install:
    pip install transformers torch
"""

from __future__ import annotations

from .base import BaselineModel


class QwenDeidBaseline(BaselineModel):
    """Qwen3-0.6B causal LM that generates <phi_entities> XML directly.

    Unlike the token-classification baselines, the model produces our schema XML
    in one shot, so predict_xml() is overridden and predict_entities() is unused.
    """

    # The model can emit every schema entity, so keep the default supported_entities.

    def __init__(
        self,
        model_src: str = "Qwen/Qwen3-0.6B-Base",
        name: str = "Qwen3-0.6B-Base (zero-shot)",
        max_new_tokens: int = 2048,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.stop_str = "</phi_entities>"
        self.max_new_tokens = max_new_tokens
        self._torch = torch

        # float16 is unstable on MPS/CPU — use float32 there, bf16 on CUDA.
        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float32
        else:
            self.device, dtype = "cpu", torch.float32
        print(f"Loading {model_src!r} on {self.device} ({dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_src, dtype=dtype, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def predict_entities(self, note: str) -> dict[str, list[str]]:
        # Not used: the model produces XML directly (predict_xml is overridden).
        raise NotImplementedError("Qwen emits XML directly; use predict_xml().")

    def predict_xml(self, note: str) -> str:
        from pipeline.eval.inference import _build_prompt

        prompt = _build_prompt(note)                       # flat-text SFT template
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=4096).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                           # greedy / deterministic
                stop_strings=self.stop_str,                # halt as soon as the tag closes
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
        xml = "<phi_entities>\n" + gen
        return xml.split(self.stop_str)[0] + self.stop_str
