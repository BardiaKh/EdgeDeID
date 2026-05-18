"""DPO scaffolding for future preference-optimization experiments."""

from __future__ import annotations

from typing import Any

from .config import DPOConfig
from .data import PreferenceExample


def prepare_dpo_dataset(examples: list[PreferenceExample]) -> Any:
    """Convert preference examples into a DPO-ready dataset object."""

    raise NotImplementedError("DPO dataset preparation will be added later.")


def build_dpo_trainer(model: Any, tokenizer: Any, config: DPOConfig, dataset: Any) -> Any:
    """Build and configure the DPO trainer."""

    raise NotImplementedError("DPO trainer construction will be added later.")


def train_dpo(model: Any, tokenizer: Any, config: DPOConfig, dataset: Any) -> Any:
    """Run a full DPO training job."""

    raise NotImplementedError("DPO training entrypoint will be added later.")
