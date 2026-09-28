#!/usr/bin/env python3
"""Assemble per-system predictions on the radiology benchmark into one wide CSV.

Output columns: ``id, text, ground_truth, <one column per system>`` — the same
shape as ``merged_all_models.csv`` (MIMIC), so ``scripts/eval_qc_verifier.py``
treats both test sets identically.

Notes and ground truth come from any ablation prediction file (they all carry
``note`` and ``GT-XML`` in benchmark order); system predictions from the
baselines notebook's ``pred_<System>.csv`` files (one ``prediction`` column,
same row order). Row alignment was checked on the last run: our SFT and GRPO
outputs are identical row-by-row across the two sources.

Usage
-----
    python scripts/make_wide_predictions.py \
        --notes-from ablation/B_predictions.csv \
        --pred-dir artifacts/baselines/predictions \
        --out artifacts/qc_eval/radiology_wide.csv
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes-from", required=True,
                    help="any ablation *_predictions.csv (columns note, GT-XML)")
    ap.add_argument("--pred-dir", required=True, help="directory of pred_<System>.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.notes_from)
    for col in ("note", "GT-XML"):
        if col not in base.columns:
            raise SystemExit(f"{args.notes_from}: missing column {col!r}")
    wide = pd.DataFrame({"id": range(len(base)),
                         "text": base["note"].fillna("").astype(str),
                         "ground_truth": base["GT-XML"].fillna("").astype(str)})

    files = sorted(glob.glob(os.path.join(args.pred_dir, "pred_*.csv")))
    if not files:
        raise SystemExit(f"no pred_*.csv in {args.pred_dir}")
    for f in files:
        name = os.path.basename(f)[len("pred_"):-len(".csv")]
        p = pd.read_csv(f)
        if len(p) != len(wide):
            raise SystemExit(f"{f}: {len(p)} rows but {len(wide)} notes — not aligned")
        wide[name] = p["prediction"].fillna("").astype(str).values

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wide.to_csv(args.out, index=False)
    print(f"{len(wide)} notes x {len(files)} systems -> {args.out}")


if __name__ == "__main__":
    main()
