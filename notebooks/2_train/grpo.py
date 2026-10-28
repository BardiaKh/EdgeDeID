# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GRPO Training — PHI De-identification
#
# Trains the SFT-initialised Qwen3-0.6B **base** model with Group Relative Policy
# Optimization (GRPO). Reward is fully computational — no LLM judge, no GPU verifier.
#
# Data: `EdgeDeID/synthetic-dataset` (`train.csv` / `test.csv`), streamed.
# Columns: `original` (note) · `anonymized` (ground-truth `<phi_entities>` XML) · `examples`.
# The prompt reproduces the **SFT flat-text template** (not a chat template) plus a
# mandatory `<qc_status>` self-assessment rule.
#
# **Reward cascade (sequential gate):**
# 1. `structure`      — valid `<phi_entities>` XML? (hard gate → 0 if it fails)
# 2. `presence`       — do tagged entities appear verbatim in the source note? (anti-hallucination)
# 3. `completeness`   — entity-level **F-β (β=2, recall-weighted)** vs ground truth; soft-match has a length guard against 1-char hacks
# 4. `qc_calibration` — **asymmetric**: confidently claiming `complete` while PHI was missed (leakage) scores 0; over-cautious `incomplete` scores 0.5
# 5. `length_penalty` — penalise redundant / over-long outputs (token-estimated)

# %%
# %pip install trl

# %%
import os
import sys
from pathlib import Path

# Set this one line when moving between local / Kaggle / Colab mounts.
PROJECT_ROOT_HINT = None  # e.g. '/kaggle/input/datasets/sinadaneshgar/edgedeid' or '/content/drive/MyDrive/EdgeDeID'
if PROJECT_ROOT_HINT is None:
    PROJECT_ROOT_HINT = Path.cwd().parent if Path.cwd().name == "RL" else Path.cwd()
PROJECT_ROOT_HINT = Path(PROJECT_ROOT_HINT).expanduser().resolve()
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

from edgedeid.runtime import detect_runtime_paths

RUNTIME = detect_runtime_paths(project_root=PROJECT_ROOT_HINT)
REPO_ROOT = RUNTIME["project_root"]
OUTPUT_ROOT = RUNTIME["output_root"]
print("runtime:", RUNTIME["env"])
print("REPO_ROOT:", REPO_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)

# load optional HF credentials from .env (HF_TOKEN); the dataset and models are public
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
HF_TOKEN = os.environ.get("HF_TOKEN")
print("HF_TOKEN loaded:", bool(HF_TOKEN))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from edgedeid.rl import GRPOConfig
from edgedeid.rl.data import prepare_grpo_split
from edgedeid.rl.grpo import build_grpo_trainer, prepare_grpo_reward_function
from edgedeid.rl.evaluation import evaluate_policy, compute_metrics
from edgedeid.training import expand_tokenizer

print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

# %% [markdown]
# ## 1 — Configuration
#
# Adjust paths and hyper-parameters here.

# %%
# ── model ──────────────────────────────────────────────────────────────────
SFT_MODEL_ID   = "EdgeDeID/Qwen3-0.6B-Base_SFT"      # HF repo of the SFT checkpoint

# ── dataset ────────────────────────────────────────────────────────────────
DATASET_ID     = "EdgeDeID/synthetic-dataset"        # HF dataset repo
TRAIN_FILE     = "train.csv"
EVAL_FILE      = "test.csv"                           # set None to skip eval
NOTE_KEY       = "original"                           # raw note column
GT_KEY         = "anonymized"                         # ground-truth <phi_entities> XML
EXAMPLES_KEY   = "examples"                           # "NONE" or JSON few-shot list

# ── runtime overrides (env) ──────────────────────────────────────────────────
# Defaults reproduce the 1000-step pilot. For a full epoch over the 25,972-row
# train split: GRPO_PDTBS=8 GRPO_GRAD_ACCUM=8 (-> 8 prompts/optimizer-step) and
# GRPO_MAX_STEPS=3247. pdtbs=8 peaks ~62GB on the longest prompts (fits 96GB);
# run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
RUN_NAME       = os.environ.get("GRPO_RUN_NAME", "grpo-edgedeid-qwen3-0.6b")
PDTBS          = int(os.environ.get("GRPO_PDTBS", "1"))
GRAD_ACCUM     = int(os.environ.get("GRPO_GRAD_ACCUM", "8"))
MAX_STEPS      = int(os.environ.get("GRPO_MAX_STEPS", "1000"))
OUTPUT_DIR     = str(OUTPUT_ROOT / "checkpoints" / RUN_NAME)

# ── training ───────────────────────────────────────────────────────────────
config = GRPOConfig(
    model_name              = SFT_MODEL_ID,
    output_dir              = OUTPUT_DIR,
    run_name                = RUN_NAME,
    wandb_project           = "huggingface",    # logs to wandb.ai/bardiakh/huggingface
    learning_rate           = 1e-5,
    per_device_train_batch_size = PDTBS,
    per_device_eval_batch_size  = 1,
    num_generations         = 8,                # completions per prompt per step
    gradient_accumulation_steps = GRAD_ACCUM,
    logging_steps           = 1,                # log every step to keep stdout lively!
    max_steps               = MAX_STEPS,        # required: streaming has no len()
    max_prompt_length       = None,             # None = no truncation; our SFT prompts run 1000-4600+ tokens and the instructions sit at the front (TRL truncates from the left)
    max_completion_length   = 1024,
    seed                    = 2025,
    beta                    = float(os.environ.get("GRPO_BETA", "0.04")),  # KL to SFT ref; 0 = none (all runs before 2026-10-02)
    reward_weights          = {                 # weights for combined reward
        "presence":     0.30,
        "completeness": 0.40,                   # recall-weighted F-beta (beta=2)
        "qc":           0.15,                   # asymmetric calibration
        "length":       0.15,
    },
)

print(config)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% [markdown]
# ## 2 — Load Model & Tokenizer
#
# Loads the base model into memory, forcing `float16` to ensure compatibility with GPUs that do not support `bfloat16` natively.
# This step will also optionally wrap the model with **PEFT/LoRA** for memory-efficient training if `USE_LORA` is enabled.

# %%
# ── 1. Load Model & Tokenizer ──────────────────────────────────────────────
device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Loading {SFT_MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_ID, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_ID,
    dtype=torch.bfloat16,  # bf16: full-param GRPO is numerically unstable in fp16 (policy collapses after the first update); Blackwell supports bf16 natively
    device_map=device_map,
    token=HF_TOKEN
)
expand_tokenizer(tokenizer, model)
print("Model loaded successfully.")

# ── 2. PEFT / LoRA (Optional) ──────────────────────────────────────────────
USE_LORA = False   # set True on low-VRAM machines

if USE_LORA:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
else:
    print("Running in full-parameter mode (No LoRA).")

# %% [markdown]
# ## 3 — Prepare Dataset
#
# Downloads and loads the synthetic training dataset directly from the Hugging Face Hub. 
# Using `load_dataset` without streaming ensures the dataset is downloaded into memory completely (which `trl`'s `GRPOTrainer` requires). The row mapping formats the input features according to the configured `GRPOConfig`.

# %%
from huggingface_hub import hf_hub_download

# resolve the files from the Hub, then load the CSV.
train_path = hf_hub_download(
    repo_id=DATASET_ID, filename=TRAIN_FILE, repo_type="dataset", token=HF_TOKEN
)
data_files = {"train": train_path}
if EVAL_FILE:
    data_files["test"] = hf_hub_download(
        repo_id=DATASET_ID, filename=EVAL_FILE, repo_type="dataset", token=HF_TOKEN
    )

raw = load_dataset("csv", data_files=data_files)
print(raw)
print("columns:", raw["train"].column_names)

# %%
# map raw rows -> {prompt, source_note, ground_truth}.
# the GRPO prompt reproduces the SFT flat-text template (+ <qc_status> rule),
# NOT a chat template — the SFT checkpoint is a base model.
train_dataset = prepare_grpo_split(
    raw["train"],
    note_key=NOTE_KEY,
    ground_truth_key=GT_KEY,
    examples_key=EXAMPLES_KEY,
)

eval_dataset = None
if EVAL_FILE and "test" in raw:
    eval_dataset = prepare_grpo_split(
        raw["test"],
        note_key=NOTE_KEY,
        ground_truth_key=GT_KEY,
        examples_key=EXAMPLES_KEY,
    )

sample0 = train_dataset[0]
print("keys:", list(sample0.keys()))
print("\n--- prompt (head) ---\n", sample0["prompt"][:600])
print("\n--- ground_truth (head) ---\n", sample0["ground_truth"][:300])

# %% [markdown]
# ## 4 — Sanity-check Reward Functions
#
# Run the reward pipeline on a single example before launching training.

# %%
from edgedeid.rl.rewards import (
    reward_xml_structure,
    reward_phi_text_presence,
    reward_completeness,
    reward_qc_calibration,
    reward_length_penalty,
    compute_grpo_reward,
)

sample = train_dataset[0]
note   = sample["source_note"]
gt     = sample["ground_truth"]

# use the ground truth itself as a perfect-score completion
fake_completion = gt + "\n<qc_status>complete</qc_status>"

c = reward_completeness(fake_completion, gt)
print("structure   :", reward_xml_structure(fake_completion))
print("presence    :", reward_phi_text_presence(fake_completion, note))
print("completeness:", c, "(F-beta, beta=2 recall-weighted)")
print("qc          :", reward_qc_calibration(fake_completion, c), "(asymmetric)")
print("length      :", reward_length_penalty(fake_completion))
print("combined    :", compute_grpo_reward(fake_completion, note, gt))

# negative control: a degenerate 1-char hack should NOT score well
hack = "<phi_entities><names><name id='1' type='patient'><variations>" \
       "<value>a</value></variations></name></names></phi_entities>" \
       "<qc_status>complete</qc_status>"
print("\nhack completeness:", reward_completeness(hack, gt), "(expect ~0, length guard)")

# %% [markdown]
# ## 5 — Build Trainer & Train
#
# Initialize Weights & Biases (if configured) and build the GRPO Trainer using the compiled reward functions and datasets. The trainer uses `trl` under the hood to manage generation and trajectory-based optimisation.

# %%
import wandb
if config.wandb_project:
    wandb.init(project=config.wandb_project, name=config.run_name, entity="bardiakh")

# %%
from transformers import TrainerCallback

class LiveConsoleLoggingCallback(TrainerCallback):
    """Custom callback to print live metrics cleanly when not using wandb"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Filter and format logs beautifully
            metrics = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in logs.items() if k not in ["epoch", "step"])
            print(f"Step [{state.global_step}/{args.max_steps}] -> {metrics}")


# %%
reward_fn = prepare_grpo_reward_function(config)

trainer = build_grpo_trainer(
    model        = model,
    tokenizer    = tokenizer,
    config       = config,
    reward_fn    = reward_fn,
    train_dataset= train_dataset,
    eval_dataset = eval_dataset,
)

# Inject our lively stdout callback
trainer.add_callback(LiveConsoleLoggingCallback())

# Resume: GRPO_RESUME=1 continues from the newest checkpoint-* in OUTPUT_DIR (if any),
# so a crash or preemption on a shared GPU costs at most save_steps (500) steps.
from transformers.trainer_utils import get_last_checkpoint
RESUME = None
if os.environ.get("GRPO_RESUME") == "1" and os.path.isdir(OUTPUT_DIR):
    RESUME = get_last_checkpoint(OUTPUT_DIR)
print("Pushed trainer logic... launching training!", f"(resuming from {RESUME})" if RESUME else "")
trainer.train(resume_from_checkpoint=RESUME)

# %% [markdown]
# ## 6 — Evaluate
#
# Runs a quick evaluation pipeline (using temperature inference and token batching) on the offline `eval_dataset` test split to estimate structural fidelity and F-beta scores post-RL.

# %%
if eval_dataset is not None:
    metrics = evaluate_policy(
        model     = trainer.model,
        tokenizer = tokenizer,
        dataset   = eval_dataset,
        max_new_tokens = 1024,
        temperature    = 0.0,
        batch_size     = 4,
    )
    print("Eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
else:
    print("No eval split — skipping evaluation.")

# %% [markdown]
# ## 7 — Save & Push to HuggingFace
#
# Commits the final model weights and tokenizer configurations back to local disk and pushes them to your targeted Hugging Face Hub repository.

# %%
HF_REPO_ID = "EdgeDeID/Qwen3-0.6B-GRPO-KL"   # target HF repo

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

# push to HF Hub — gated behind PUSH_TO_HUB=1 so a test run never auto-publishes
if os.environ.get("PUSH_TO_HUB") == "1":
    trainer.model.push_to_hub(HF_REPO_ID)
    tokenizer.push_to_hub(HF_REPO_ID)
    print(f"Pushed to {HF_REPO_ID}")
else:
    print("Skipping HF push (set PUSH_TO_HUB=1 to enable).")

# %%
if config.wandb_project:
    wandb.finish()
