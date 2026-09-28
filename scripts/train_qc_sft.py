#!/usr/bin/env python3
"""Stage 1 of the QC verifier: supervised LoRA adapter on the extraction model.

Starts from the existing extraction SFT checkpoint and trains a LoRA adapter on
the verification examples built by ``scripts/build_qc_dataset.py``. Loss is on
the completion only — the verdict (``complete</qc_status>`` or
``incomplete</qc_status>``), plus the XML for the small share of single-pass
examples — so a deliberately corrupted candidate in the prompt is never a
training target and cannot teach the model to omit entities.

Why an adapter: extraction runs with the adapter OFF (weights identical to the
checkpoint whose numbers are in the paper); verification runs with it ON. The
extraction results therefore cannot regress, and the verifier is a separate,
swappable artifact that can audit any system's output.

This stage is what makes stage 2 (GRPO) possible at all: GRPO sharpens
behaviour the policy already produces. Here the verdict is put in-support, and
the logged ``val_auroc`` is the "SFT-only" baseline GRPO must beat.

Usage
-----
    python scripts/train_qc_sft.py --data-dir data/qc --out-dir checkpoints/qc_sft
    # smaller / faster:
    python scripts/train_qc_sft.py --data-dir data/qc --out-dir checkpoints/qc_sft \
        --max-train 3000 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.qc.train_utils import build_config, load_base, load_env, val_auroc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="EdgeDeID/Qwen3-0.6B-Base_SFT")
    ap.add_argument("--data-dir", default="data/qc")
    ap.add_argument("--out-dir", default="checkpoints/qc_sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=8192,
                    help="must exceed the longest prompt: truncation would cut the "
                         "instruction block or the candidate XML")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=None, help="cap examples (smoke tests)")
    ap.add_argument("--attn", default=None, help="e.g. flash_attention_2 (default: sdpa)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--push-to-hub", default=None, help="HF repo id for the adapter")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("HF_TOKEN")

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    train_rows = pd.read_json(os.path.join(args.data_dir, "qc_sft_train.jsonl"),
                              lines=True).to_dict("records")
    val_rows = pd.read_json(os.path.join(args.data_dir, "qc_sft_val.jsonl"),
                            lines=True).to_dict("records")
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    # prompt/completion only: TRL treats every other string column as unknown
    train_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                                  for r in train_rows])
    print(f"train examples: {len(train_ds)} | val rows: {len(val_rows)}")

    model, tok = load_base(args.base, dtype=args.dtype, attn=args.attn, token=token)
    tok.padding_side = "right"
    model.config.use_cache = False                     # gradient checkpointing

    # Length guard: silently truncated prompts would train on a note without
    # its candidate XML or its instructions.
    longest = max(len(tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"])
                  for r in train_rows[:2000])
    print(f"longest prompt+completion (first 2000 rows): {longest} tokens")
    if longest >= args.max_length:
        raise SystemExit(f"--max-length {args.max_length} would truncate examples "
                         f"({longest} tokens); raise it")

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    use_bf16 = args.dtype == "bfloat16" and torch.cuda.is_available()
    cfg = build_config(
        SFTConfig,
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,              # loss on the verdict, never the candidate
        logging_steps=10,
        save_strategy="no",
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)

    before = val_auroc(model, tok, val_rows, limit=300)
    print("before training (untrained adapter ~ extraction model as zero-shot verifier):", before)
    trainer.train()
    after = val_auroc(model, tok, val_rows)
    print("after SFT:", after)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "qc_sft_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "val_before": before, "val_after": after}, f, indent=2)
    print(f"saved adapter -> {args.out_dir}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, token=token, private=True)
        tok.push_to_hub(args.push_to_hub, token=token, private=True)
        print(f"pushed -> {args.push_to_hub}")


if __name__ == "__main__":
    main()
