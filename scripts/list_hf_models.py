#!/usr/bin/env python3
"""
List all models (and datasets) in the EdgeDeID HuggingFace organization.

Usage:
    python scripts/list_hf_models.py

Prints a table of all repos in the EdgeDeID org and optionally saves to JSON.
Token resolution (automatic):
  1. ~/.huggingface/token (local setup)
  2. .env file (Colab/Kaggle)
  3. HF_TOKEN environment variable
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Missing dependency: huggingface-hub")
    print("Install: pip install huggingface-hub")
    sys.exit(1)

ORG = "EdgeDeID"
OUTPUT_FILE = Path(__file__).parent.parent / "hf_collection_info.json"


def load_token():
    hf_token_path = Path.home() / ".huggingface" / "token"
    if hf_token_path.exists():
        return hf_token_path.read_text().strip()
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        match = re.search(r"HF_TOKEN\s*=\s*(.+)", env_path.read_text())
        if match:
            return match.group(1).strip()
    return os.environ.get("HF_TOKEN")


def list_org_repos(api: HfApi):
    print(f"\n{'='*70}")
    print(f"  EdgeDeID HuggingFace Organization — All Repos")
    print(f"{'='*70}\n")

    # ── Models ────────────────────────────────────────────────────────────────
    models = list(api.list_models(author=ORG))
    print(f"📦 Models ({len(models)} found)")
    print(f"  {'Repo ID':<45} {'Downloads':>10}  {'Last modified'}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*20}")
    models_info = []
    for m in sorted(models, key=lambda x: x.id):
        last_mod = str(m.last_modified)[:10] if m.last_modified else "—"
        downloads = m.downloads if m.downloads is not None else 0
        print(f"  {m.id:<45} {downloads:>10,}  {last_mod}")
        models_info.append({
            "id": m.id,
            "pipeline_tag": m.pipeline_tag,
            "tags": list(m.tags or []),
            "downloads": downloads,
            "likes": m.likes,
            "last_modified": str(m.last_modified),
            "private": m.private,
        })

    # ── Datasets ──────────────────────────────────────────────────────────────
    datasets = list(api.list_datasets(author=ORG))
    print(f"\n🗂  Datasets ({len(datasets)} found)")
    print(f"  {'Repo ID':<45} {'Downloads':>10}  {'Last modified'}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*20}")
    datasets_info = []
    for d in sorted(datasets, key=lambda x: x.id):
        last_mod = str(d.last_modified)[:10] if d.last_modified else "—"
        downloads = d.downloads if d.downloads is not None else 0
        print(f"  {d.id:<45} {downloads:>10,}  {last_mod}")
        datasets_info.append({
            "id": d.id,
            "tags": list(d.tags or []),
            "downloads": downloads,
            "likes": d.likes,
            "last_modified": str(d.last_modified),
            "private": d.private,
        })

    # ── Spaces ────────────────────────────────────────────────────────────────
    spaces = list(api.list_spaces(author=ORG))
    if spaces:
        print(f"\n🚀 Spaces ({len(spaces)} found)")
        for s in sorted(spaces, key=lambda x: x.id):
            print(f"  {s.id}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Total: {len(models)} model(s), {len(datasets)} dataset(s), {len(spaces)} space(s)")
    print(f"{'='*70}\n")

    return {"models": models_info, "datasets": datasets_info}


if __name__ == "__main__":
    token = load_token()
    if token:
        print("✓ HF token loaded")
    else:
        print("ℹ️  No token found — public repos only")

    api = HfApi(token=token)
    info = list_org_repos(api)

    OUTPUT_FILE.write_text(json.dumps(info, indent=2))
    print(f"💾 Saved to {OUTPUT_FILE.relative_to(Path(__file__).parent.parent)}")
