"""Evaluation helpers for GRPO-trained PHI de-identification models."""

from __future__ import annotations

from typing import Any, Iterable

from .rewards import (
    compute_grpo_reward,
    reward_completeness,
    reward_phi_text_presence,
    reward_xml_structure,
)


def compute_metrics(
    completions: list[str],
    source_notes: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """Aggregate reward metrics over a batch of model completions.

    Returns mean values for: structure, presence, completeness, combined.
    """
    n = len(completions)
    if n == 0:
        return {}

    struct_scores, presence_scores, complete_scores, combined_scores = [], [], [], []
    for c, note, gt in zip(completions, source_notes, ground_truths):
        struct_scores.append(reward_xml_structure(c))
        presence_scores.append(reward_phi_text_presence(c, note))
        complete_scores.append(reward_completeness(c, gt))
        combined_scores.append(compute_grpo_reward(c, note, gt))

    return {
        "structure": sum(struct_scores) / n,
        "presence": sum(presence_scores) / n,
        "completeness": sum(complete_scores) / n,
        "combined": sum(combined_scores) / n,
    }


def evaluate_policy(
    model: Any,
    tokenizer: Any,
    dataset: Iterable[dict[str, Any]],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    batch_size: int = 4,
    qc_config: Any | None = None,
) -> dict[str, float]:
    """Generate completions for every item in dataset and return aggregate metrics.

    dataset items must have keys: 'prompt', 'source_note', 'ground_truth'.
    'prompt' is a flat text string (the SFT base model has no chat template),
    so it is tokenised directly rather than via apply_chat_template.

    If ``qc_config`` is a ``pipeline.qc.QCConfig`` the layered QC pipeline is
    applied to every completion before scoring (ablation rows D / F): the
    intrinsic <qc_status> is read from the completion, deterministic repair +
    detector reconciliation enforce the recall floor, and escalation is gated.
    """
    import torch

    rows = list(dataset)
    completions: list[str] = []
    source_notes: list[str] = []
    ground_truths: list[str] = []

    model.eval()
    # Decoder-only batched generation requires LEFT padding: with the default
    # right padding, pad tokens land between the prompt and the generated text
    # for shorter prompts in a batch, corrupting their completions (and the
    # prompt_len slice). Also disable truncation — prompts (instructions+schema
    # +few-shot) must stay intact, exactly as in training (max_prompt_length=None).
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                prompts = [r["prompt"] for r in batch]
                source_notes.extend(r["source_note"] for r in batch)
                ground_truths.extend(r["ground_truth"] for r in batch)

                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(model.device)

                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                prompt_len = encoded["input_ids"].shape[1]
                for seq in out:
                    completions.append(
                        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    )
    finally:
        tokenizer.padding_side = prev_side

    if qc_config is not None:
        from ..qc import apply_qc_batch

        completions, _ = apply_qc_batch(completions, source_notes, qc_config)

    return compute_metrics(completions, source_notes, ground_truths)


def evaluate_preference_accuracy(model: Any, dataset: Iterable[Any]) -> float:
    """TODO (Phase 2 / DPO): Measure how often the policy prefers the better response."""
    raise NotImplementedError("Preference accuracy evaluation — Phase 2.")
