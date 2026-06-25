"""Headless evaluation of ONE vLLM-served Qwen checkpoint on en_data.csv.

Mirrors §5 of baselines.ipynb but runnable from the shell, so it can be driven by
run_our_vllm.sh to evaluate all three checkpoints unattended. Assumes a vLLM server
is already running for `--model` on `--port` (use serve_vllm.sh).

Appends one row to results/comparison_ours.csv and writes per-model predictions.

Example:
    python eval_one_vllm.py --model EdgeDeID/Qwen3-0.6B-Base_SFT --name EdgeDeID-SFT --port 9995
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
REPO_DIR = HERE.parents[1]                      # scripts/ -> repo root
sys.path.insert(0, str(REPO_DIR))

from openai import OpenAI

from edgedeid.data.preprocessing import preprocess_text
from edgedeid.data.xml_utils import convert_to_xml
from edgedeid.eval import batch_inference, compute_metrics, print_report

QWEN_PARAMS_M = 596.0                            # Qwen3-0.6B parameter count (millions)


def load_en_data() -> tuple[list[str], list[str]]:
    df = pd.read_csv(REPO_DIR / "evals" / "ext_eval" / "en_data.csv")
    df = preprocess_text(df, input_col="Report", output_col="DeID")
    df["GT-XML"] = df["DeID"].apply(convert_to_xml)
    for tag in ["<anonymized_report>", "</anonymized_report>", "<anonymized_report/>",
                "<raw_report>", "</raw_report>", "<raw_report/>"]:
        df["Report"] = df["Report"].str.replace(tag, "", regex=False).str.strip()
    return df["Report"].tolist(), df["GT-XML"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vLLM --model id (must match the running server)")
    ap.add_argument("--name", required=True, help="display name for the results row")
    ap.add_argument("--port", type=int, default=9995)
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--out", default=str(REPO_DIR / "artifacts" / "baselines"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    notes, gt_xmls = load_en_data()
    print(f"Loaded {len(notes)} samples (en_data.csv)")

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1")

    t0 = time.perf_counter()
    preds = batch_inference(notes, client, args.model,
                            temperature=0.0, max_tokens=2048, max_workers=args.max_workers)
    sec_per_note = (time.perf_counter() - t0) / max(len(notes), 1)

    r = compute_metrics(preds, gt_xmls, source_notes=notes)   # ours emits the full schema
    print_report(r, title=args.name)

    pd.Series(preds, name="prediction").to_csv(out_dir / "predictions" / f"pred_{args.name}.csv", index=False)

    row = {
        "Model": args.name,
        "Micro-P": round(r.micro_precision, 4), "Micro-R": round(r.micro_recall, 4),
        "Micro-F1": round(r.micro_f1, 4), "F-b(2)": round(r.f_beta, 4),
        "Macro-F1": round(r.macro_f1, 4), "Zero-leak": round(r.zero_leak_rate, 4),
        "XML Parse": round(r.xml_parse_rate, 4), "Halluc.": round(r.hallucination_rate, 4),
        "Token Reduction": (round(r.token_reduction_rate, 4)
                            if r.token_reduction_rate == r.token_reduction_rate else None),
        "#Params (M)": QWEN_PARAMS_M, "Sec/Note": round(sec_per_note, 4),
    }
    csv_path = out_dir / "comparison_ours.csv"
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    df = df[df.get("Model") != args.name] if "Model" in df.columns else df   # idempotent
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"\nAppended {args.name} to {csv_path}  ({sec_per_note*1000:.1f} ms/note)")


if __name__ == "__main__":
    main()
