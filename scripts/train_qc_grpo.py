#!/usr/bin/env python3
"""Stage 2 of the QC verifier: GRPO on the verdict, starting from the SFT adapter.

The policy sees ``instructions + note + candidate XML + <qc_status>`` and writes
only the verdict (a handful of tokens), so rollouts are cheap. Reward
(``edgedeid.qc.verifier.qc_verifier_reward``):

    well-formed verdict      +0.2
    correct                  +0.8   (complete/complete, incomplete/incomplete)
    over-cautious            +0.4   (said incomplete, was complete)
    overconfident            +0.0   (said complete, PHI was missed)
    malformed                 0     (total)

so correct 1.0 > over-cautious 0.6 > overconfident 0.2 > malformed 0. The 0.5
calibration value for over-caution puts the reward-optimal flagging threshold at
P(incomplete) > 1/3.

Why this can learn where the first GRPO run could not: every rollout contains a
verdict, and after stage 1 both verdicts carry real probability, so groups have
within-group reward variance from the first step. The script logs the share of
zero-variance groups when TRL reports it; if it stays near 1, sampling is too
cold or stage 1 collapsed to one verdict — raise --temperature.

KL: default beta=0 (no reference model). With a LoRA policy TRL's reference is
the model with the adapter disabled — the plain extraction model, which cannot
verify at all — so a KL term would pull the verifier back toward chance. A low
learning rate and a few hundred steps keep the update conservative instead.

Usage
-----
    python scripts/train_qc_grpo.py --data-dir data/qc --sft-adapter checkpoints/qc_sft \
        --out-dir checkpoints/qc_grpo --max-steps 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.qc.train_utils import build_config, load_base, load_env, val_auroc  # noqa: E402
from edgedeid.qc.verifier import qc_verifier_reward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="EdgeDeID/Qwen3-0.6B-Base_SFT")
    ap.add_argument("--sft-adapter", required=True, help="output dir of train_qc_sft.py")
    ap.add_argument("--data-dir", default="data/qc")
    ap.add_argument("--out-dir", default="checkpoints/qc_grpo")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=1,
                    help="unique prompts per device step (x num_generations completions)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient (see docstring)")
    ap.add_argument("--max-completion-length", type=int, default=8,
                    help="'incomplete</qc_status>' is 6 tokens; the rest is headroom")
    ap.add_argument("--attn", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--push-to-hub", default=None)
    args = ap.parse_args()

    load_env()
    token = os.environ.get("HF_TOKEN")

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    rows = pd.read_json(os.path.join(args.data_dir, "qc_grpo_train.jsonl"),
                        lines=True).to_dict("records")
    val_rows = pd.read_json(os.path.join(args.data_dir, "qc_sft_val.jsonl"),
                            lines=True).to_dict("records")
    train_ds = Dataset.from_list([{"prompt": r["prompt"], "is_complete": bool(r["is_complete"])}
                                  for r in rows])
    inc = sum(not r["is_complete"] for r in rows) / len(rows)
    print(f"GRPO prompts: {len(train_ds)}  (incomplete {inc:.1%})")

    base, tok = load_base(args.base, dtype=args.dtype, attn=args.attn, token=token)
    tok.padding_side = "left"                       # generation
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()

    before = val_auroc(model, tok, val_rows)
    print("before GRPO (= SFT adapter):", before)

    use_bf16 = args.dtype == "bfloat16" and torch.cuda.is_available()
    cfg = build_config(
        GRPOConfig,
        output_dir=args.out_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.num_generations * args.prompts_per_step,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # trl<1.0 only: its default of 512 left-truncates the ~1.5-4k-token
        # prompts, deleting the instruction block. None disables truncation.
        max_prompt_length=None,
        temperature=args.temperature,
        beta=args.beta,
        remove_unused_columns=False,                # the reward reads `is_complete`
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        report_to=args.report_to,
        log_completions=False,
        seed=args.seed,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[qc_verifier_reward], args=cfg,
                          train_dataset=train_ds, processing_class=tok)
    trainer.train()

    after = val_auroc(trainer.model, tok, val_rows)
    print("after GRPO:", after)

    os.makedirs(args.out_dir, exist_ok=True)
    trainer.model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    hist = [h for h in trainer.state.log_history if "reward" in h or "rewards/mean" in h]
    with open(os.path.join(args.out_dir, "qc_grpo_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "val_before": before, "val_after": after,
                   "log_tail": hist[-10:]}, f, indent=2, default=str)
    print(f"saved adapter -> {args.out_dir}")

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, token=token, private=True)
        tok.push_to_hub(args.push_to_hub, token=token, private=True)
        print(f"pushed -> {args.push_to_hub}")


if __name__ == "__main__":
    main()
