"""GRPO training for PHI de-identification policy optimization."""

from __future__ import annotations

from typing import Any, Callable

from .config import GRPOConfig
from .rewards import make_reward_fn


def prepare_grpo_reward_function(
    config: GRPOConfig,
    *,
    source_notes: list[str] | None = None,
    ground_truths: list[str] | None = None,
) -> Callable[..., list[float]]:
    """Build the composite reward callback for TRL GRPOTrainer.

    When source_notes / ground_truths are None the function reads them from
    kwargs (i.e., from dataset columns named 'source_note' / 'ground_truth').
    """
    return make_reward_fn(
        source_notes=source_notes,
        ground_truths=ground_truths,
        weights=config.reward_weights,
    )


def build_grpo_trainer(
    model: Any,
    tokenizer: Any,
    config: GRPOConfig,
    reward_fn: Callable[..., list[float]],
    train_dataset: Any,
    eval_dataset: Any | None = None,
) -> Any:
    """Configure and return a TRL GRPOTrainer.

    Parameters
    ----------
    model        : PreTrainedModel (already loaded + optionally wrapped for PEFT)
    tokenizer    : PreTrainedTokenizer / PreTrainedTokenizerFast
    config       : GRPOConfig — our wrapper; TRL GRPOConfig is built internally
    reward_fn    : (completions: list[str], **kwargs) -> list[float]
    train_dataset: datasets.Dataset with 'prompt', 'source_note', 'ground_truth' columns
    eval_dataset : optional validation split
    """
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.grpo_config import GRPOConfig as TRLGRPOConfig
    
    trl_cfg = TRLGRPOConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        seed=config.seed,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        # CRITICAL: forward max_prompt_length. TRL's default is 512 and our flat
        # SFT prompts (instructions + schema + few-shot examples) run 1000-4600+
        # tokens. TRL left-truncates the prompt, so any cap below the full length
        # drops the leading <instructions> block — the policy then never learns to
        # emit <phi_entities> and every completion fails the structure gate
        # (reward 0, zero variance, zero gradient). None = no truncation.
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.wandb_project else "none",
    )

    return GRPOTrainer(
        model=model,
        args=trl_cfg,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def train_grpo(
    model: Any,
    tokenizer: Any,
    config: GRPOConfig,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    *,
    source_notes: list[str] | None = None,
    ground_truths: list[str] | None = None,
) -> Any:
    """End-to-end GRPO training entry point.

    Wires the reward function and trainer together and runs training.
    Returns the trained model.
    """
    reward_fn = prepare_grpo_reward_function(
        config,
        source_notes=source_notes,
        ground_truths=ground_truths,
    )
    trainer = build_grpo_trainer(
        model, tokenizer, config, reward_fn, train_dataset, eval_dataset
    )
    trainer.train()
    return trainer.model
