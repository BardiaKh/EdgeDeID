"""RL training helpers for GRPO policy optimization."""

from .config import DPOConfig, GRPOConfig, RLConfig
from .data import (
    GRPO_SYSTEM_PROMPT,
    PreferenceExample,
    build_grpo_dataset,
    format_preference_prompt,
    grpo_map_fn,
    prepare_grpo_split,
)
from .evaluation import compute_metrics, evaluate_policy, evaluate_preference_accuracy
from .grpo import build_grpo_trainer, prepare_grpo_reward_function, train_grpo
from .rewards import (
    QC_COMPLETE_THRESHOLD,
    compute_grpo_reward,
    entity_recall,
    make_reward_fn,
    reward_completeness,
    reward_length_penalty,
    reward_phi_text_presence,
    reward_qc_calibration,
    reward_xml_structure,
)