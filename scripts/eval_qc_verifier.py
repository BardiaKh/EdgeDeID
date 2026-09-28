#!/usr/bin/env python3
"""Evaluate QC-verifier configurations as a human-review filter.

For every (note, system output) pair in each test set, the verifier gives
P(incomplete); the label is whether that output actually leaks (zero-leak rule,
``edgedeid.eval.metrics.note_is_complete``). Positive class = incomplete.

Configurations (``--config name=adapter_dir``; empty dir = no adapter):
  zero_shot=        the extraction model itself, prompted as a verifier
  sft=<dir>         after stage 1 (scripts/train_qc_sft.py)
  grpo=<dir>        after stage 2 (scripts/train_qc_grpo.py)
The sft -> grpo difference is the RL contribution, measured, not assumed.

Test sets (``--testset name=wide.csv``), each ``id, text, ground_truth, <systems>``:
  radiology   from scripts/make_wide_predictions.py
  mimic       merged_all_models.csv
Scoring outputs from *other* systems is the "someone else de-identified this"
test; ``--self-system`` (default EdgeDeID-SFT) is reported separately because a
verifier sharing the extractor's weights may share its blind spots.

Thresholds are chosen on the synthetic VALIDATION split only and applied
unchanged to the test sets:
  1/3          the operating point the GRPO reward optimizes for
  0.5          plain majority
  val@catch95  highest threshold catching >=95% of incomplete validation notes

Outputs (--out-dir):
  scores.csv      one row per (config, testset, system, note): label, p_incomplete
  summary.csv     AUROC, AUPRC, ECE, Brier, and flag/catch/residual-leak at each
                  threshold, for all systems pooled, self, others, and per system
  curves.csv      review-budget curves (flag rate vs residual leak / catch rate)
  thresholds.json thresholds picked on validation, per config

Usage
-----
    python scripts/eval_qc_verifier.py \
        --config zero_shot= --config sft=checkpoints/qc_sft --config grpo=checkpoints/qc_grpo \
        --testset radiology=artifacts/qc_eval/radiology_wide.csv \
        --testset mimic=merged_all_models.csv \
        --val data/qc/qc_sft_val.jsonl --out-dir artifacts/qc_eval
    # faster: --max-per-system 400 subsamples MIMIC notes per system
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.qc.train_utils import load_base, load_env  # noqa: E402
from edgedeid.qc.verifier import build_verifier_prompt, label_candidate, score_incomplete  # noqa: E402
from edgedeid.qc.verifier_eval import budget_curve, summarize, threshold_for_catch_rate  # noqa: E402

META = ("id", "text", "ground_truth")


def parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_pairs(path: str, systems: list[str] | None, max_per_system: int | None, seed: int):
    """(system, row, prompt, label) for every system column of a wide CSV."""
    import pandas as pd

    df = pd.read_csv(path)
    missing = [c for c in META if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    cols = [c for c in df.columns if c not in META]
    if systems:
        cols = [c for c in cols if c in systems]
    rows = list(range(len(df)))
    if max_per_system and max_per_system < len(rows):
        rows = sorted(random.Random(seed).sample(rows, max_per_system))
    notes = df["text"].fillna("").astype(str)
    golds = df["ground_truth"].fillna("").astype(str)
    out = []
    for sys_name in cols:
        cands = df[sys_name].fillna("").astype(str)
        for r in rows:
            out.append((sys_name, r, build_verifier_prompt(notes[r], cands[r]),
                        label_candidate(cands[r], golds[r])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="EdgeDeID/Qwen3-0.6B-Base_SFT")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--testset", action="append", required=True)
    ap.add_argument("--val", default="data/qc/qc_sft_val.jsonl")
    ap.add_argument("--val-limit", type=int, default=800)
    ap.add_argument("--systems", default=None, help="comma-separated subset of system columns")
    ap.add_argument("--self-system", default="EdgeDeID-SFT")
    ap.add_argument("--max-per-system", type=int, default=None)
    ap.add_argument("--catch-target", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out-dir", default="artifacts/qc_eval")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("HF_TOKEN")

    import pandas as pd
    import torch

    configs = parse_kv(args.config)
    testsets = parse_kv(args.testset)
    systems = [s.strip() for s in args.systems.split(",")] if args.systems else None
    dtype = args.dtype if torch.cuda.is_available() else "float32"

    val = pd.read_json(args.val, lines=True)
    if "kind" in val:
        val = val[val["kind"] == "prefill"]
    val = val.head(args.val_limit)
    val_y = [0 if c else 1 for c in val["is_complete"]]

    pairs = {name: load_pairs(path, systems, args.max_per_system, args.seed)
             for name, path in testsets.items()}
    for name, ps in pairs.items():
        n_inc = sum(not lab for *_, lab in ps)
        print(f"testset {name}: {len(ps)} pairs, {n_inc/len(ps):.1%} incomplete")

    os.makedirs(args.out_dir, exist_ok=True)
    score_rows, summary_rows, curve_rows, thresholds_all = [], [], [], {}

    for cfg_name, adapter in configs.items():
        print(f"\n=== config: {cfg_name}  (adapter: {adapter or 'none'})")
        model, tok = load_base(args.base, dtype=dtype, attn=args.attn, token=token)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        vp = score_incomplete(model, tok, val["prompt"].tolist(), batch_size=args.batch_size)
        th = {"1/3": 1 / 3, "0.5": 0.5,
              f"val@catch{int(args.catch_target * 100)}":
                  threshold_for_catch_rate(val_y, vp, args.catch_target)}
        thresholds_all[cfg_name] = th
        vs = summarize(val_y, vp, th)
        print(f"  synthetic val: AUROC={vs['auroc']}  thresholds={th}")
        summary_rows.append({"config": cfg_name, "testset": "synthetic_val", "group": "all", **vs})

        for ts_name, ps in pairs.items():
            p = score_incomplete(model, tok, [x[2] for x in ps], batch_size=args.batch_size)
            for (sys_name, r, _, lab), pi in zip(ps, p):
                score_rows.append({"config": cfg_name, "testset": ts_name, "system": sys_name,
                                   "row": r, "is_complete": lab, "p_incomplete": pi})
            y = [0 if lab else 1 for *_, lab in ps]
            groups = {"all": list(range(len(ps))),
                      "self": [i for i, x in enumerate(ps) if x[0] == args.self_system],
                      "others": [i for i, x in enumerate(ps) if x[0] != args.self_system]}
            for s in sorted({x[0] for x in ps}):
                groups[f"system:{s}"] = [i for i, x in enumerate(ps) if x[0] == s]
            for g, idx in groups.items():
                if not idx:
                    continue
                yy, pp = [y[i] for i in idx], [p[i] for i in idx]
                summary_rows.append({"config": cfg_name, "testset": ts_name, "group": g,
                                     **summarize(yy, pp, th)})
                if g in ("all", "self", "others"):
                    for pt in budget_curve(yy, pp):
                        curve_rows.append({"config": cfg_name, "testset": ts_name,
                                           "group": g, **pt})
            s_all = next(r for r in reversed(summary_rows)
                         if r["testset"] == ts_name and r["group"] == "all")
            key = next(k for k in th if k.startswith("val@"))
            print(f"  {ts_name:10s} AUROC={s_all['auroc']}  ECE={s_all['ece']:.3f}  "
                  f"@{key}: review {s_all[f'flag_rate@{key}']:.1%}, "
                  f"catch {s_all[f'catch_rate@{key}']:.1%}, residual leak "
                  f"{s_all[f'residual_leak_rate@{key}']:.2%} "
                  f"(no filter: {s_all['base_leak_rate']:.2%})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(score_rows).to_csv(os.path.join(args.out_dir, "scores.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(args.out_dir, "curves.csv"), index=False)
    with open(os.path.join(args.out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds_all, f, indent=2)
    print(f"\nsaved -> {args.out_dir}/ (scores.csv, summary.csv, curves.csv, thresholds.json)")


if __name__ == "__main__":
    main()
