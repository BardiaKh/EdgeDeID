#!/usr/bin/env python
"""Run an EdgeDeID checkpoint over the MIMIC notes with a few-shot example set.

Same notes, batching and greedy decoding as the zero-shot MIMIC rows (batch 16
in file order, plain SFT prompt, 1,024 new tokens); the only change is the
``<examples>`` block. Writes id,<name> to --out; merge into the gold-scored wide
CSV and run scripts/score_mimic.py on it.

    python scripts/mimic_fewshot.py --model SFT/weights/Qwen3-0.6B-Base_SFT \
        --name EdgeDeID-SFT-3shot --out artifacts/fewshot/pred_sft.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from edgedeid.baselines import QwenDeidBaseline  # noqa: E402
from edgedeid.eval.inference import _build_prompt  # noqa: E402
from edgedeid.fewshot import load_examples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", default="icu_nursing")
    ap.add_argument("--notes", default="artifacts/mimic_gold/merged_all_models_gold.csv")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.notes)
    notes = [s.strip() for s in df.text.fillna("")]
    examples = load_examples(args.examples)
    m = QwenDeidBaseline(args.model, name=args.name, examples=examples)

    # The baseline truncates prompts at 4,096 tokens from the RIGHT, which
    # would cut the note's end and the <phi_entities> opener. Refuse instead.
    longest = max(len(m.tokenizer(_build_prompt(n, examples)).input_ids) for n in notes)
    print(f"{len(notes)} notes, {len(examples)} examples, longest prompt {longest} tokens")
    if longest > 4096:
        raise SystemExit("a prompt exceeds the 4,096-token truncation limit")

    preds: list[str] = []
    for i in tqdm(range(0, len(notes), args.batch), desc=args.name):
        preds += m.predict_xml_batch(notes[i:i + args.batch])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], args.name: preds}).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
