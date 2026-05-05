#!/usr/bin/env python3
"""
Fetch detailed information about HuggingFace models and datasets.
"""
from huggingface_hub import model_info, dataset_info
import json

# Model and Dataset IDs
MODELS = [
    "EdgeDeID/Qwen3-1.7B-Base_SFT",
    "EdgeDeID/Qwen3-0.6B-Base_SFT",
]

DATASETS = [
    "EdgeDeID/synthetic-dataset",
]

def get_model_details(model_id):
    """Fetch model details from HuggingFace Hub."""
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
            "files": [f.filename for f in info.siblings[:10]],  # First 10 files
        }
    except Exception as e:
        return {"id": model_id, "error": str(e)}

def get_dataset_details(dataset_id):
    """Fetch dataset details from HuggingFace Hub."""
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
    print("=" * 80)
    print("EDGDEID HUGGINGFACE COLLECTION - DETAILED INFORMATION")
    print("=" * 80)
    
    print("\n📦 MODELS:")
    print("-" * 80)
    models_info = {}
    for model_id in MODELS:
        print(f"\nFetching: {model_id}...")
        info = get_model_details(model_id)
        models_info[model_id] = info
        print(json.dumps(info, indent=2))
    
    print("\n" + "=" * 80)
    print("\n📊 DATASETS:")
    print("-" * 80)
    datasets_info = {}
    for dataset_id in DATASETS:
        print(f"\nFetching: {dataset_id}...")
        info = get_dataset_details(dataset_id)
        datasets_info[dataset_id] = info
        print(json.dumps(info, indent=2))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Successfully fetched information for {len(models_info)} models and {len(datasets_info)} dataset(s)")
    
    # Save to JSON for reference
    all_info = {
        "models": models_info,
        "datasets": datasets_info,
    }
    
    with open("hf_collection_info.json", "w") as f:
        json.dump(all_info, f, indent=2)
    print("✅ Saved detailed information to hf_collection_info.json")
