"""Inference helpers for evaluation — runs a vLLM-served model via OpenAI API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..prompts import SYSTEM_PROMPT


def _build_prompt(note: str, examples: list[dict] | None = None) -> str:
    """Build the raw completion prompt used by eval notebooks."""
    examples_str = ""
    if examples:
        parts = [
            f"<example_note>\n{ex['original']}\n</example_note>\n\n"
            f"<example_phi_entities>\n{ex['anonymized']}\n</example_phi_entities>"
            for ex in examples
        ]
        examples_str = "<examples>\n" + "\n\n".join(parts) + "\n</examples>\n\n"

    return (
        f"<instructions>\n{SYSTEM_PROMPT}\n</instructions>\n\n"
        f"{examples_str}"
        f"<note>\n{note.strip()}\n</note>\n\n"
        "<phi_entities>\n"
    )


def run_vllm_model(
    note: str,
    client: Any,
    model_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: int = 2025,
    examples: list[dict] | None = None,
) -> str:
    """Send one note to a vLLM-served model and return the full phi_entities XML."""
    prompt = _build_prompt(note, examples)
    response = client.completions.create(
        model=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        stop=["</phi_entities>"],
    )
    text = "<phi_entities>\n" + response.choices[0].text
    text = text.split("</phi_entities>")[0] + "</phi_entities>"
    return text


def batch_inference(
    notes: list[str],
    client: Any,
    model_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: int = 2025,
    max_workers: int = 32,
    examples: list[dict] | None = None,
) -> list[str]:
    """Run inference on a list of notes using a thread-pool (mirrors eval notebooks).

    Returns predictions in the same order as the input notes.
    """
    from tqdm import tqdm

    results = [""] * len(notes)

    def _process(i: int) -> tuple[int, str]:
        return i, run_vllm_model(
            notes[i], client, model_id,
            temperature=temperature, max_tokens=max_tokens,
            seed=seed, examples=examples,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, i) for i in range(len(notes))]
        for future in tqdm(as_completed(futures), total=len(notes), desc="Inference"):
            i, result = future.result()
            results[i] = result

    return results
