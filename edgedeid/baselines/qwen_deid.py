"""Causal-LM PHI de-identification baseline (Qwen3-0.6B family).

Wraps a Qwen3-0.6B causal LM that emits <phi_entities> XML directly via the SFT
prompt template. Parameterized by checkpoint + display name so one class covers:
  * our fine-tuned student : EdgeDeID/Qwen3-0.6B-Base_SFT
  * the un-finetuned base   : Qwen/Qwen3-0.6B-Base  (zero-shot reference)

Install:
    pip install transformers torch
"""

from __future__ import annotations

import re

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
        # 1024 exceeds the longest reference output (568 Qwen tokens on the test
        # set) by ~1.8x, so valid completions are never truncated, while keeping
        # the budget uniform across all generative models for a fair comparison.
        max_new_tokens: int = 1024,
        emit_qc_status: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.stop_str = "</phi_entities>"
        # GRPO checkpoints emit `<qc_status>…</qc_status>` on the line AFTER
        # </phi_entities> (see notebooks/2_train/grpo.py). Halting generation at
        # </phi_entities> — and then hard-truncating there in _wrap() — makes
        # that tag structurally impossible to observe, which is what drove the
        # tag-coverage=0.00 result. When measuring the metacognitive signal, keep
        # generating past the XML and stop on the qc_status tag instead.
        self.emit_qc_status = emit_qc_status
        self.gen_stop_str = "</qc_status>" if emit_qc_status else self.stop_str
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

    def _wrap(self, gen: str) -> str:
        """Re-attach the opening tag and trim everything after the closing tag.

        With ``emit_qc_status``, the trailing ``<qc_status>`` tag is preserved
        rather than truncated, since it is the signal being measured.
        """
        xml = "<phi_entities>\n" + gen
        head = xml.split(self.stop_str)[0] + self.stop_str
        if not self.emit_qc_status:
            return head
        tail = xml.split(self.stop_str, 1)[1] if self.stop_str in xml else ""
        m = re.search(r"<qc_status\s*>(complete|incomplete)</qc_status>", tail, re.I)
        return head + ("\n" + m.group(0) if m else "")

    def predict_xml(self, note: str) -> str:
        from edgedeid.eval.inference import _build_prompt

        prompt = _build_prompt(note, with_qc_status=self.emit_qc_status)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=4096).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                           # greedy / deterministic
                stop_strings=self.gen_stop_str,            # halt once the final tag closes
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                    skip_special_tokens=True)
        return self._wrap(gen)

    def predict_xml_batch(self, notes: list[str]) -> list[str]:
        """Generate XML for a batch of notes in one forward pass (GPU-friendly).

        Left-pads the prompts so the generated suffix aligns across the batch,
        then slices each completion off after the shared prompt length.
        """
        from edgedeid.eval.inference import _build_prompt

        prompts = [_build_prompt(n, with_qc_status=self.emit_qc_status) for n in notes]
        # Left-padding is required for correct batched decoding of causal LMs.
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True,
                                max_length=4096, padding=True).to(self.device)
        self.tokenizer.padding_side = prev_side
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stop_strings=self.gen_stop_str,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]          # shared after left-pad
        gens = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        return [self._wrap(g) for g in gens]

    def sample_xml_batch(
        self, notes: list[str], k: int = 5, temperature: float = 0.7,
    ) -> list[list[str]]:
        """Draw ``k`` sampled XML completions per note (do_sample, temperature > 0).

        Returns ``[[xml_1, …, xml_k], …]`` — one inner list per input note — for
        the K-sample *union* decoding arm. ``num_return_sequences`` gets all K
        draws of a note in a single generate() call. Use ``predict_xml_batch``
        for the greedy (temperature 0) arm; sampling with k draws only makes
        sense when temperature > 0.
        """
        from edgedeid.eval.inference import _build_prompt

        prompts = [_build_prompt(n, with_qc_status=self.emit_qc_status) for n in notes]
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True,
                                max_length=4096, padding=True).to(self.device)
        self.tokenizer.padding_side = prev_side
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=k,
                stop_strings=self.gen_stop_str,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        gens = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        wrapped = [self._wrap(g) for g in gens]
        # generate() returns sequences grouped per input: note_i occupies the
        # rows [i*k : (i+1)*k].
        return [wrapped[i * k:(i + 1) * k] for i in range(len(notes))]
