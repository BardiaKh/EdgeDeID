"""Configuration objects for RL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RLConfig:
    """Shared settings for RL runs."""

    model_name: str
    output_dir: str
    max_prompt_length: int = 4096
    max_completion_length: int = 8192
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 1e-5  # GRPO on a 0.6B policy
    seed: int = 2025
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DPOConfig(RLConfig):
    """Configuration for DPO-style preference optimization."""

    beta: float = 0.1
    max_pairs_per_prompt: int = 1
    reference_model_name: str | None = None
    loss_types: list = field(default_factory=list)
    loss_weights: list = field(default_factory=list)


@dataclass(slots=True)
class GRPOConfig(RLConfig):
    """Configuration for GRPO-style policy optimization."""

    num_generations: int = 4
    group_size: int = 4
    # streaming IterableDataset has no __len__, so max_steps must be set
    max_steps: int = 1000
    gradient_accumulation_steps: int = 8
    logging_steps: int = 5
    reward_clip_min: float | None = None
    reward_clip_max: float | None = None
    # reward weighting for compute_grpo_reward (presence / completeness / qc / length)
    reward_weights: dict | None = None
    # wandb project name; set to None to disable wandb logging
    wandb_project: str | None = None
    # KL coefficient against the frozen SFT reference. This field did not exist
    # before 2026-10-02 and build_grpo_trainer never set it, so every GRPO run got
    # TRL's default beta=0.0 (confirmed in all three local runs' training_args.bin)
    # -- no anchor at all on output the reward does not score. The published
    # checkpoint drifted on exactly that: it writes the wrong ISO year in every
    # <formatted> date (235/235 on the radiology benchmark; SFT 0/235). 0.04 is the
    # GRPO paper's value and TRL's own default before it moved to 0.0.
    beta: float = 0.04
