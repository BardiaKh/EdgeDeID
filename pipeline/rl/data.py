"""GRPO / preference data helpers for PHI de-identification training."""

from __future__ import annotations

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


# ─── GRPO data helpers ────────────────────────────────────────────────────────

def format_preference_prompt(
    note: str,
    instruction: str | None = None,
) -> list[dict[str, str]]:
    """Format a medical note as a chat message list for GRPO/DPO pipelines.

    Returns a list compatible with tokenizer.apply_chat_template().
    instruction defaults to GRPO_SYSTEM_PROMPT (includes the <qc_status> requirement).
    """
    system = instruction if instruction is not None else GRPO_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": note},
    ]


def build_grpo_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    note_key: str = "text",
    ground_truth_key: str = "phi_entities",
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Convert tabular rows into GRPO-ready records.

    Each record contains:
      - prompt       : chat messages for tokenizer.apply_chat_template()
      - source_note  : raw medical note (passed to reward functions as dataset column)
      - ground_truth : expected XML output (passed to reward functions as dataset column)
    """
    records = []
    for row in rows:
        note = row[note_key]
        gt = row[ground_truth_key]
        records.append({
            "prompt": format_preference_prompt(note, instruction),
            "source_note": note,
            "ground_truth": gt,
        })
    return records


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
