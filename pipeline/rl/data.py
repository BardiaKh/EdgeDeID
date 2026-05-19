"""GRPO / preference data helpers for PHI de-identification training.

The synthetic dataset (``EdgeDeID/synthetic-dataset``) ships ``train.csv`` and
``test.csv`` with three columns:

* ``original``   — raw medical note (model input)
* ``anonymized`` — the ground-truth ``<phi_entities>`` XML block (model target)
* ``examples``   — ``"NONE"`` or a JSON list of {original, anonymized} few-shot examples

The SFT checkpoint (``EdgeDeID/Qwen3-0.6B-Base_SFT``) is a *base* model that was
fine-tuned on a flat text template (NOT a chat template):

    <instructions>
    {SYSTEM_PROMPT}
    </instructions>

    {examples_str}<note>
    {original}
    </note>

GRPO must reproduce that exact prompt template so the policy stays in the
distribution it was SFT'd on; the only change is swapping in
``GRPO_SYSTEM_PROMPT`` (which appends the mandatory ``<qc_status>`` rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..prompts import SYSTEM_PROMPT

# GRPO extends the base prompt with a mandatory <qc_status> self-assessment tag.
GRPO_SYSTEM_PROMPT: str = SYSTEM_PROMPT + (
    "\n\n8. After the <phi_entities> block, add a self-assessment on a new line:\n"
    "   <qc_status>complete</qc_status>   — if you believe you found all PHI entities.\n"
    "   <qc_status>incomplete</qc_status> — if you think some PHI entities may be missing.\n"
    "   This tag is mandatory."
)


@dataclass(slots=True)
class PreferenceExample:
    """A prompt with preferred and rejected model responses (used by DPO)."""

    prompt: str
    preferred: str
    rejected: str
    metadata: dict[str, Any] | None = None


# ─── prompt construction (must mirror SFT/sft.ipynb create_instruction) ────────

def _build_examples_str(examples_field: Any) -> str:
    """Render the few-shot examples block exactly as the SFT template did.

    ``examples_field`` is the raw ``examples`` column value: the literal string
    ``"NONE"`` or a JSON-encoded list of {original, anonymized} dicts.
    """
    if examples_field is None or examples_field == "NONE":
        return ""
    try:
        examples = json.loads(examples_field)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not examples:
        return ""
    blocks = [
        f"<example_note>\n{ex['original']}\n</example_note>\n\n"
        f"<example_phi_entities>\n{ex['anonymized']}\n</example_phi_entities>"
        for ex in examples
    ]
    return f"<examples>\n{chr(10).join(blocks)}\n</examples>\n\n"


def format_preference_prompt(
    note: str,
    instruction: str | None = None,
    examples_field: Any = "NONE",
) -> str:
    """Build the flat-text GRPO/SFT prompt string for one note.

    Returns a plain string (NOT chat messages) because the SFT base model was
    trained without a chat template. ``instruction`` defaults to
    ``GRPO_SYSTEM_PROMPT`` (includes the <qc_status> requirement).
    """
    system = instruction if instruction is not None else GRPO_SYSTEM_PROMPT
    examples_str = _build_examples_str(examples_field)
    return (
        f"<instructions>\n{system}\n</instructions>\n\n"
        f"{examples_str}<note>\n{note}\n</note>\n\n"
    )


# ─── GRPO data helpers ────────────────────────────────────────────────────────

def _to_grpo_record(
    row: dict[str, Any],
    *,
    note_key: str,
    ground_truth_key: str,
    examples_key: str,
    instruction: str | None,
) -> dict[str, Any]:
    note = row[note_key]
    return {
        "prompt": format_preference_prompt(
            note, instruction, row.get(examples_key, "NONE")
        ),
        "source_note": note,
        "ground_truth": row[ground_truth_key],
    }


def grpo_map_fn(
    *,
    note_key: str = "original",
    ground_truth_key: str = "anonymized",
    examples_key: str = "examples",
    instruction: str | None = None,
):
    """Return a ``datasets.Dataset.map``-compatible transform.

    Works for both in-memory ``Dataset`` and streaming ``IterableDataset`` —
    pass it to ``.map(grpo_map_fn(), remove_columns=[...])``. Each example is
    converted to the GRPO record schema: ``prompt`` / ``source_note`` /
    ``ground_truth``.
    """

    def _fn(row: dict[str, Any]) -> dict[str, Any]:
        return _to_grpo_record(
            row,
            note_key=note_key,
            ground_truth_key=ground_truth_key,
            examples_key=examples_key,
            instruction=instruction,
        )

    return _fn


def prepare_grpo_split(
    split: Any,
    *,
    note_key: str = "original",
    ground_truth_key: str = "anonymized",
    examples_key: str = "examples",
    instruction: str | None = None,
):
    """Map a HF dataset split (streaming or not) into GRPO-ready records.

    Preserves streaming: if ``split`` is an ``IterableDataset`` the result is
    still an ``IterableDataset`` (lazy, no full materialisation).
    """
    remove = [c for c in (note_key, ground_truth_key, examples_key)]
    # IterableDataset.map ignores num_proc; Dataset.map accepts it.
    try:
        return split.map(
            grpo_map_fn(
                note_key=note_key,
                ground_truth_key=ground_truth_key,
                examples_key=examples_key,
                instruction=instruction,
            ),
            remove_columns=remove,
        )
    except (AttributeError, TypeError):
        # plain iterable of dicts → list of records
        return build_grpo_dataset(
            split,
            note_key=note_key,
            ground_truth_key=ground_truth_key,
            examples_key=examples_key,
            instruction=instruction,
        )


def build_grpo_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    note_key: str = "original",
    ground_truth_key: str = "anonymized",
    examples_key: str = "examples",
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Eagerly convert tabular rows into GRPO-ready records (non-streaming).

    Each record contains:
      - prompt       : flat SFT-style prompt string (apply_chat_template NOT used)
      - source_note  : raw medical note (reward kwarg via dataset column)
      - ground_truth : expected <phi_entities> XML (reward kwarg via dataset column)
    """
    return [
        _to_grpo_record(
            row,
            note_key=note_key,
            ground_truth_key=ground_truth_key,
            examples_key=examples_key,
            instruction=instruction,
        )
        for row in rows
    ]


# ─── DPO helpers (Phase 2 — TODO) ────────────────────────────────────────────

def build_preference_examples(
    rows: Iterable[dict[str, Any]],
    *,
    prompt_key: str = "prompt",
    preferred_key: str = "preferred",
    rejected_key: str = "rejected",
) -> list[PreferenceExample]:
    """TODO (Phase 2): Convert tabular rows into DPO preference examples."""
    raise NotImplementedError("DPO preference example construction — Phase 2.")


def load_preference_examples(path: str | Path) -> list[PreferenceExample]:
    """TODO (Phase 2): Load DPO preference examples from disk."""
    raise NotImplementedError("DPO preference dataset loading — Phase 2.")
