#!/usr/bin/env python3
"""Score every model's MIMIC predictions under the project's eval protocol.

Input is a single wide CSV: columns ``id``, ``text``, ``ground_truth``, then one
column per model holding that model's ``<phi_entities>`` prediction. This is the
shape produced by running the baseline harness over ``mimic_data.csv`` (see
``scripts/convert_mimic_i2b2.py`` for how the ground truth is built).

Two tables are written:

* ``mimic_master_table.csv`` -- scored as-is.
* ``mimic_master_table_coverage_adjusted.csv`` -- email/url/device stripped from
  BOTH predictions and ground truth first.

The second table exists because ``compute_metrics``' ``supported_entities``
argument scopes only MACRO-F1; micro precision/recall stay full-coverage by
design, so a model is otherwise charged for predicting categories MIMIC never
annotates (it contains zero emails, urls and devices). Stripping them from both
sides is the like-for-like comparison.

Note that ``zero_leak_rate`` is reported but is NOT comparable to the in-domain
benchmark: ~60% of MIMIC records contain no PHI at all, so the metric is
trivially satisfied on those. It is deliberately omitted from the paper's MIMIC
table.

Usage
-----
    python scripts/score_mimic.py --predictions merged_all_models.csv
    python scripts/score_mimic.py --predictions merged_all_models.csv \
        --out-dir artifacts/mimic --bootstrap EdgeDeID-SFT,RoBERTa-i2b2
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.eval.metrics import bootstrap_ci, compute_metrics  # noqa: E402

# The seven categories the MIMIC corpus actually exercises. It carries zero
# emails/urls/devices and a single `id`, so macro-averaging over all ten would
# score "never had a chance to be right" as "failed".
MIMIC_ENTITIES = {"name", "age", "date", "place", "address", "phone", "id"}
ABSENT_PARENT_TAGS = ["emails", "urls", "devices"]

META_COLUMNS = ("id", "text", "ground_truth")


def strip_absent_categories(xml: str) -> str:
    """Drop the parent blocks for categories MIMIC never annotates."""
    for tag in ABSENT_PARENT_TAGS:
        xml = re.sub(rf"<{tag}\s*/>", "", xml, flags=re.I)
        xml = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", xml, flags=re.I | re.S)
    return xml


def _row(name: str, r) -> dict:
    return {
        "Model": name,
        "Micro-P": round(r.micro_precision, 4),
        "Micro-R": round(r.micro_recall, 4),
        "Micro-F1": round(r.micro_f1, 4),
        "F-b(2)": round(r.f_beta, 4),
        "Macro-F1": round(r.macro_f1, 4),
        "Zero-leak": round(r.zero_leak_rate, 4),
        "XML Parse": round(r.xml_parse_rate, 4),
        "Halluc.": round(r.hallucination_rate, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="merged_all_models.csv",
                    help="wide CSV: id,text,ground_truth,<one column per model>")
    ap.add_argument("--out-dir", default="artifacts/mimic")
    ap.add_argument("--bootstrap", default="",
                    help="comma-separated model columns to add 95%% CIs for "
                         "(slow: n_boot resamples each)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.predictions} is missing required column(s): {missing}")

    notes = df["text"].fillna("").astype(str).tolist()
    gts_raw = df["ground_truth"].fillna("").astype(str).tolist()
    gts_adj = [strip_absent_categories(g) for g in gts_raw]
    model_cols = [c for c in df.columns if c not in META_COLUMNS]
    if not model_cols:
        raise SystemExit("no model prediction columns found")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(df)} MIMIC records | {len(model_cols)} models")
    print(f"restricted supported_entities = {sorted(MIMIC_ENTITIES)}\n")

    raw_rows, adj_rows, per_entity = [], [], {}
    for col in model_cols:
        preds_raw = df[col].fillna("").astype(str).tolist()
        preds_adj = [strip_absent_categories(p) for p in preds_raw]

        r_raw = compute_metrics(preds_raw, gts_raw, source_notes=notes,
                                supported_entities=MIMIC_ENTITIES)
        r_adj = compute_metrics(preds_adj, gts_adj, source_notes=notes,
                                supported_entities=MIMIC_ENTITIES)
        raw_rows.append(_row(col, r_raw))
        adj_rows.append(_row(col, r_adj))
        per_entity[col] = {k: round(m.f1, 4) for k, m in r_adj.per_entity.items()}
        print(f"  {col:26s} F1={r_adj.micro_f1:.4f}  R={r_adj.micro_recall:.4f}  "
              f"P={r_adj.micro_precision:.4f}")

    raw = pd.DataFrame(raw_rows).sort_values("Micro-F1", ascending=False)
    adj = pd.DataFrame(adj_rows).sort_values("Micro-F1", ascending=False)
    pe = pd.DataFrame(per_entity)

    raw.to_csv(os.path.join(args.out_dir, "mimic_master_table.csv"), index=False)
    adj.to_csv(os.path.join(args.out_dir,
                            "mimic_master_table_coverage_adjusted.csv"), index=False)
    pe.to_csv(os.path.join(args.out_dir, "mimic_per_entity.csv"))

    print("\n=== COVERAGE-ADJUSTED (the table to quote) ===")
    print(adj.to_string(index=False))
    print("\n=== per-entity F1 (coverage-adjusted) ===")
    print(pe.to_string())

    for col in [c.strip() for c in args.bootstrap.split(",") if c.strip()]:
        if col not in model_cols:
            print(f"\n[bootstrap] unknown column {col!r} -- skipped")
            continue
        preds = [strip_absent_categories(p)
                 for p in df[col].fillna("").astype(str)]
        print(f"\n[bootstrap] {col} (95% CI, {args.n_boot} resamples)")
        for metric in ("micro_f1", "micro_recall", "micro_precision"):
            pt, lo, hi = bootstrap_ci(preds, gts_adj, notes,
                                      supported_entities=MIMIC_ENTITIES,
                                      metric=metric, n_boot=args.n_boot)
            print(f"    {metric:16s} {pt:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\nsaved -> {args.out_dir}/")


if __name__ == "__main__":
    main()
