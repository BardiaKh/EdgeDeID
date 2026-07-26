#!/usr/bin/env python3
"""Verify that all EdgeDeID models and datasets are accessible on HuggingFace Hub."""

import sys
from pathlib import Path

try:
    from huggingface_hub import list_repo_files, model_info, dataset_info
    from transformers import AutoTokenizer, AutoConfig
    from datasets import load_dataset, get_dataset_config_names
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install huggingface-hub transformers datasets")
    sys.exit(1)


MODELS = {
    "Qwen3-1.7B-Base_SFT": "EdgeDeID/Qwen3-1.7B-Base_SFT",
    "Qwen3-0.6B-Base_SFT": "EdgeDeID/Qwen3-0.6B-Base_SFT",
    "Qwen3-0.6B-GRPO": "EdgeDeID/Qwen3-0.6B-GRPO",
}

DATASETS = {
    "synthetic-dataset": "EdgeDeID/synthetic-dataset",
}


def check_model(name, repo_id):
    print(f"Checking model: {name}  ({repo_id})")
    print("-" * 70)
    try:
        info = model_info(repo_id)
        print(f"  Created:       {info.created_at}")
        print(f"  Last modified: {info.last_modified}")
        print(f"  Downloads:     {info.downloads:,}")
        print(f"  Private:       {info.private}")
        print(f"  Tags:          {', '.join(info.tags) if info.tags else 'none'}")

        files = list(list_repo_files(repo_id, repo_type="model"))
        ext_counts: dict[str, int] = {}
        for f in files:
            ext = Path(f).suffix or "(none)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        print(f"  Files ({len(files)} total):", ", ".join(f"{e}: {n}" for e, n in sorted(ext_counts.items())))

        config = AutoConfig.from_pretrained(repo_id)
        print(f"  Architecture:  {config.model_type}")
        print(f"  Hidden size:   {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  Vocab size:    {getattr(config, 'vocab_size', 'N/A')}")

        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        print(f"  Tokenizer:     {type(tokenizer).__name__}, vocab={len(tokenizer)}")

        print("  PASS\n")
        return True
    except Exception as e:
        print(f"  FAIL: {str(e)[:200]}\n")
        return False


def check_dataset(name, repo_id):
    print(f"Checking dataset: {name}  ({repo_id})")
    print("-" * 70)
    try:
        info = dataset_info(repo_id)
        print(f"  Created:       {info.created_at}")
        print(f"  Last modified: {info.last_modified}")
        print(f"  Downloads:     {info.downloads:,}")
        print(f"  Private:       {info.private}")
        print(f"  Tags:          {', '.join(info.tags) if info.tags else 'none'}")

        try:
            configs = get_dataset_config_names(repo_id)
            print(f"  Configs:       {', '.join(configs) if configs else 'default'}")
        except Exception:
            pass

        ds = load_dataset(repo_id, split=None)
        for split_name, split_data in ds.items():
            print(f"  Split '{split_name}': {len(split_data)} rows, columns: {', '.join(split_data.column_names)}")

        print("  PASS\n")
        return True
    except Exception as e:
        print(f"  FAIL: {str(e)[:200]}\n")
        return False


def main():
    print("\nEdgeDeID HuggingFace Collection — Verification")
    print("=" * 70)

    results: dict[str, bool] = {}

    print("\n--- Models ---\n")
    for name, repo_id in MODELS.items():
        results[name] = check_model(name, repo_id)

    print("\n--- Datasets ---\n")
    for name, repo_id in DATASETS.items():
        results[name] = check_dataset(name, repo_id)

    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} checks passed")

    if passed < len(results):
        print("If repos are private, run: huggingface-cli login")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
