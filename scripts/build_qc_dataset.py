#!/usr/bin/env python3
"""Build the QC-verifier datasets (SFT, GRPO, validation) from the synthetic
TRAINING split. CPU only, a few minutes.

Outputs (in --out-dir):
  qc_sft_train.jsonl   prompt / completion / is_complete / source / note_idx / kind
  qc_sft_val.jsonl     same schema; in-distribution validation (threshold picking)
  qc_grpo_train.jsonl  prompt / is_complete / source / note_idx  (no completion)
  stats.json           class balance per split and per perturbation family

Notes are split into disjoint pools (sft / grpo / val) before any candidate is
generated, so no training note appears in two splits. The in-domain benchmark
and MIMIC are never read here.

Usage
-----
    python scripts/build_qc_dataset.py --out-dir data/qc
    python scripts/build_qc_dataset.py --out-dir data/qc --train-csv data/train.csv \
        --extra-candidates artifacts/qc_candidates/train_candidates.csv

Class balance: SFT defaults to 30% incomplete, GRPO to 45%. Not lower —
if SFT leaves P(incomplete) near zero, most GRPO groups on truly incomplete
notes contain no "incomplete" sample and give no gradient (the zero-variance
failure this whole design exists to avoid). The deployment base rate is handled
at evaluation time by picking the threshold, not by starving training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.qc.verifier_data import (  # noqa: E402
    load_extra_candidates,
    rebalance,
    split_notes,
    synth_candidates,
    to_grpo_example,
    to_prefill_example,
    to_single_pass_example,
)

DATASET_ID = "EdgeDeID/synthetic-dataset"


def load_env(path: str = ".env") -> None:
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_train_csv(path: str | None) -> str:
    if path:
        return path
    load_env()
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATASET_ID, "train.csv", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    inc = sum(not r["is_complete"] for r in rows)
    by_src = Counter((r["source"], "complete" if r["is_complete"] else "incomplete") for r in rows)
    return {
        "n": n,
        "incomplete_frac": round(inc / n, 4) if n else None,
        "by_source": {f"{s}|{lab}": c for (s, lab), c in sorted(by_src.items())},
        "prompt_chars_median": int(pd.Series([len(r["prompt"]) for r in rows]).median()) if n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", default=None,
                    help="synthetic train.csv (default: download from HF, needs HF_TOKEN)")
    ap.add_argument("--out-dir", default="data/qc")
    ap.add_argument("--n-notes", type=int, default=8000,
                    help="training notes to draw from (of ~26k)")
    ap.add_argument("--cands-per-note", type=int, default=2)
    ap.add_argument("--split", default="sft=0.7,grpo=0.2,val=0.1",
                    help="note-pool fractions")
    ap.add_argument("--sft-incomplete-frac", type=float, default=0.30)
    ap.add_argument("--grpo-incomplete-frac", type=float, default=0.45)
    ap.add_argument("--val-incomplete-frac", type=float, default=0.30)
    ap.add_argument("--single-pass-frac", type=float, default=0.15,
                    help="extra SFT examples teaching the tag after the model's OWN "
                         "XML (single-pass mode), as a fraction of SFT prefill rows")
    ap.add_argument("--extra-candidates", default=None,
                    help="CSV of real system outputs on TRAINING notes: columns "
                         "original, candidate[, source]")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    fracs = {k: float(v) for k, v in (kv.split("=") for kv in args.split.split(","))}
    rng = random.Random(args.seed)

    path = resolve_train_csv(args.train_csv)
    df = pd.read_csv(path)
    notes = df["original"].astype(str).tolist()
    golds = df["anonymized"].astype(str).tolist()
    print(f"train split: {len(df)} notes  ({path})")

    pools = split_notes(len(df), args.n_notes, fracs, args.seed)
    print("note pools:", {k: len(v) for k, v in pools.items()})

    cands = {name: [] for name in pools}
    for name, idx in pools.items():
        for i in idx:
            cands[name] += synth_candidates(i, notes[i], golds[i], args.cands_per_note, rng)

    if args.extra_candidates:
        gold_by_note = {n.strip(): i for i, n in enumerate(notes)}
        extra = load_extra_candidates(pd.read_csv(args.extra_candidates).to_dict("records"),
                                      gold_by_note, notes, golds)
        pool_of = {i: name for name, idx in pools.items() for i in idx}
        kept = 0
        for c in extra:                    # only notes already in a pool, so the
            if c.note_idx in pool_of:      # split stays disjoint
                cands[pool_of[c.note_idx]].append(c)
                kept += 1
        print(f"extra candidates: {len(extra)} matched a training note, {kept} fell in a pool")

    target = {"sft": args.sft_incomplete_frac, "grpo": args.grpo_incomplete_frac,
              "val": args.val_incomplete_frac}
    balanced = {name: rebalance(c, target.get(name, 0.3), rng) for name, c in cands.items()}

    sft = [to_prefill_example(c) for c in balanced.get("sft", [])]
    n_single = int(round(args.single_pass_frac * len(sft)))
    for i in rng.sample(pools.get("sft", []), min(n_single, len(pools.get("sft", [])))):
        sft.append(to_single_pass_example(i, notes[i], golds[i]))
    rng.shuffle(sft)
    val = [to_prefill_example(c) for c in balanced.get("val", [])]
    grpo = [to_grpo_example(c) for c in balanced.get("grpo", [])]

    os.makedirs(args.out_dir, exist_ok=True)
    outs = {"qc_sft_train.jsonl": sft, "qc_sft_val.jsonl": val, "qc_grpo_train.jsonl": grpo}
    for fn, rows in outs.items():
        pd.DataFrame(rows).to_json(os.path.join(args.out_dir, fn),
                                   orient="records", lines=True, force_ascii=False)

    stats = {"args": vars(args), "train_csv": path,
             "splits": {fn: summarize(rows) for fn, rows in outs.items()}}
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for fn, s in stats["splits"].items():
        print(f"\n{fn}: n={s['n']}  incomplete={s['incomplete_frac']}  "
              f"median prompt chars={s['prompt_chars_median']}")
        for k, v in s["by_source"].items():
            print(f"    {k:32s} {v}")
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
