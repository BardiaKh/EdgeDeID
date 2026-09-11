#!/usr/bin/env python3
"""Paired bootstrap comparison between two prediction sets.

``bootstrap_ci`` in ``edgedeid.eval.metrics`` gives a *marginal* interval per
model, which answers "how precise is this number?" but not the question the
ablation actually asks: "is B different from A?". Two marginal intervals can
overlap while the paired difference is reliably non-zero, so comparisons in the
paper use the difference distribution computed over the *same* resampled notes.

This is the script behind the paper's two significance claims:
  * SFT (row B) vs SFT+GRPO (row D)
  * raw GRPO (row D) vs GRPO+QC (row E)

Usage
-----
    # ablation rows, by letter (reads ablation/<KEY>_predictions.csv)
    python scripts/compare_runs.py --rows B D
    python scripts/compare_runs.py --rows D E --ablation-dir lastrun_results/ablation

    # or two explicit CSVs with note/LLM-XML/GT-XML columns
    python scripts/compare_runs.py --a path/to/B.csv --b path/to/D.csv

Reads a negative delta as "B is worse than A" on that metric. An interval that
excludes zero is reported as significant at the 5% level.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.eval.metrics import compute_metrics  # noqa: E402

METRICS = ["micro_f1", "micro_recall", "micro_precision", "f_beta",
           "zero_leak_rate", "hallucination_rate"]

# Metrics where a larger value is worse, so the "significant" verdict reads the
# right way round in the printed table.
LOWER_IS_BETTER = {"hallucination_rate"}


def load(path: str):
    df = pd.read_csv(path)
    for col in ("LLM-XML", "GT-XML"):
        if col not in df.columns:
            raise SystemExit(f"{path}: missing required column {col!r}")
    notes = (df["note"].fillna("").astype(str).tolist()
             if "note" in df.columns else None)
    return (df["LLM-XML"].fillna("").astype(str).tolist(),
            df["GT-XML"].fillna("").astype(str).tolist(),
            notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs=2, metavar=("A", "B"),
                    help="ablation row letters, e.g. --rows B D")
    ap.add_argument("--ablation-dir", default="ablation")
    ap.add_argument("--a", help="baseline predictions CSV")
    ap.add_argument("--b", help="comparison predictions CSV")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    if args.rows:
        a_path = os.path.join(args.ablation_dir, f"{args.rows[0]}_predictions.csv")
        b_path = os.path.join(args.ablation_dir, f"{args.rows[1]}_predictions.csv")
        a_name, b_name = f"row {args.rows[0]}", f"row {args.rows[1]}"
    elif args.a and args.b:
        a_path, b_path = args.a, args.b
        a_name, b_name = os.path.basename(a_path), os.path.basename(b_path)
    else:
        raise SystemExit("give either --rows A B or both --a and --b")

    for p in (a_path, b_path):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")

    a_preds, a_gts, notes = load(a_path)
    b_preds, b_gts, _ = load(b_path)
    if a_gts != b_gts:
        raise SystemExit("ground truth differs between the two runs -- "
                         "they must be scored on the same evaluation set")
    n = len(a_gts)
    print(f"paired bootstrap: ({b_name}) - ({a_name})   n={n}, "
          f"{args.n_boot} resamples, seed={args.seed}\n")

    rng = random.Random(args.seed)
    idx_sets = [[rng.randrange(n) for _ in range(n)] for _ in range(args.n_boot)]

    print(f"{'metric':20s} {'delta':>9s}  {'95% CI of delta':>22s}   verdict")
    print("-" * 74)
    for m in METRICS:
        base = (getattr(compute_metrics(b_preds, a_gts, notes), m)
                - getattr(compute_metrics(a_preds, a_gts, notes), m))
        diffs = []
        for idx in idx_sets:
            g = [a_gts[i] for i in idx]
            nt = [notes[i] for i in idx] if notes else None
            diffs.append(
                getattr(compute_metrics([b_preds[i] for i in idx], g, nt), m)
                - getattr(compute_metrics([a_preds[i] for i in idx], g, nt), m))
        diffs.sort()
        lo = diffs[int(0.025 * args.n_boot)]
        hi = diffs[int(0.975 * args.n_boot)]
        if lo > 0 or hi < 0:
            better = (base > 0) ^ (m in LOWER_IS_BETTER)
            verdict = "SIGNIFICANT " + ("(better)" if better else "(WORSE)")
        else:
            verdict = "not sig (spans 0)"
        print(f"{m:20s} {base:+9.4f}  [{lo:+.4f}, {hi:+.4f}]   {verdict}")


if __name__ == "__main__":
    main()
