#!/usr/bin/env python3
"""Fetch metadata for all EdgeDeID models and datasets from HuggingFace Hub."""

import json
from pathlib import Path
from huggingface_hub import model_info, dataset_info

MODELS = [
    "EdgeDeID/Qwen3-1.7B-Base_SFT",
    "EdgeDeID/Qwen3-0.6B-Base_SFT",
]

DATASETS = [
    "EdgeDeID/synthetic-dataset",
]

OUTPUT_FILE = Path(__file__).parent.parent / "hf_collection_info.json"


def get_model_details(model_id):
    try:
        info = model_info(model_id)
        return {
            "id": model_id,
            "pipeline_tag": info.pipeline_tag,
            "tags": info.tags,
            "created_at": str(info.created_at),
            "last_modified": str(info.last_modified),
            "downloads": info.downloads,
            "likes": info.likes,
            "private": info.private,
            "files": [f.filename for f in info.siblings[:10]],
        }
    except Exception as e:
        return {"id": model_id, "error": str(e)}


def get_dataset_details(dataset_id):
    try:
        info = dataset_info(dataset_id)
        return {
            "id": dataset_id,
            "tags": info.tags,
            "created_at": str(info.created_at),
            "last_modified": str(info.last_modified),
            "downloads": info.downloads,
            "likes": info.likes,
            "private": info.private,
        }
    except Exception as e:
        return {"id": dataset_id, "error": str(e)}


if __name__ == "__main__":
    print("=" * 70)
    print("EdgeDeID HuggingFace Collection — Metadata")
    print("=" * 70)

    models_info = {}
    print("\nModels:")
    print("-" * 70)
    for model_id in MODELS:
        print(f"  Fetching {model_id}...")
        info = get_model_details(model_id)
        models_info[model_id] = info
        print(json.dumps(info, indent=4))

    datasets_info = {}
    print("\nDatasets:")
    print("-" * 70)
    for dataset_id in DATASETS:
        print(f"  Fetching {dataset_id}...")
        info = get_dataset_details(dataset_id)
        datasets_info[dataset_id] = info
        print(json.dumps(info, indent=4))

    all_info = {"models": models_info, "datasets": datasets_info}
    OUTPUT_FILE.write_text(json.dumps(all_info, indent=2))
    print(f"\nSaved to {OUTPUT_FILE}")
