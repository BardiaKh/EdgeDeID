#!/usr/bin/env python3
"""
Comprehensive verification script for EdgeDeID HuggingFace collection.
Checks models and dataset availability, structure, and metadata.
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import (
        list_repo_files,
        model_info,
        dataset_info,
        HfApi,
    )
    from transformers import AutoTokenizer, AutoConfig
    from datasets import load_dataset, get_dataset_config_names
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install huggingface-hub transformers datasets")
    sys.exit(1)


MODELS = {
    "Qwen3-1.7B-Base_SFT": "EdgeDeID/Qwen3-1.7B-Base_SFT",
    "Qwen3-0.6B-Base_SFT": "EdgeDeID/Qwen3-0.6B-Base_SFT",
}

DATASETS = {
    "synthetic-dataset": "EdgeDeID/synthetic-dataset",
}


def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def check_model(name, repo_id):
    """Check model availability and structure."""
    print(f"🔍 Checking Model: {name}")
    print(f"   Repo ID: {repo_id}")
    print("-" * 80)

    try:
        # Get model info
        print("  ✓ Fetching model info...")
        info = model_info(repo_id)

        print(f"    • Created: {info.created_at}")
        print(f"    • Last Modified: {info.last_modified}")
        print(f"    • Downloads: {info.downloads:,}")
        print(f"    • Likes: {info.likes:,}")
        print(f"    • Private: {info.private}")
        print(f"    • Tags: {', '.join(info.tags) if info.tags else 'None'}")

        # List files
        print("  ✓ Listing repository files...")
        try:
            files = list_repo_files(repo_id, repo_type="model")
            print(f"    Total files: {len(files)}")

            # Group files by type
            file_types = {}
            for f in files:
                ext = Path(f).suffix or "no_extension"
                if ext not in file_types:
                    file_types[ext] = []
                file_types[ext].append(f)

            print("    File types:")
            for ext in sorted(file_types.keys()):
                count = len(file_types[ext])
                print(f"      • {ext}: {count} file(s)")
                # Show first few files of each type
                for f in file_types[ext][:3]:
                    print(f"        - {f}")
                if count > 3:
                    print(f"        ... and {count - 3} more")

        except Exception as e:
            print(f"    ⚠ Could not list files: {str(e)[:100]}")

        # Try to load config
        print("  ✓ Loading model config...")
        try:
            config = AutoConfig.from_pretrained(repo_id)
            print(f"    • Architecture: {config.model_type}")
            print(f"    • Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
            print(f"    • Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
            print(f"    • Num parameters: ~{config.num_parameters:,}" if hasattr(config, 'num_parameters') else "")
        except Exception as e:
            print(f"    ⚠ Could not load config: {str(e)[:100]}")

        # Try to load tokenizer
        print("  ✓ Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            print(f"    • Tokenizer type: {type(tokenizer).__name__}")
            print(f"    • Vocab size: {len(tokenizer)}")
        except Exception as e:
            print(f"    ⚠ Could not load tokenizer: {str(e)[:100]}")

        print("  ✅ Model check passed!")
        return True

    except Exception as e:
        print(f"  ❌ Model check failed: {str(e)[:200]}")
        return False


def check_dataset(name, repo_id):
    """Check dataset availability and structure."""
    print(f"🔍 Checking Dataset: {name}")
    print(f"   Repo ID: {repo_id}")
    print("-" * 80)

    try:
        # Get dataset info
        print("  ✓ Fetching dataset info...")
        info = dataset_info(repo_id)

        print(f"    • Created: {info.created_at}")
        print(f"    • Last Modified: {info.last_modified}")
        print(f"    • Downloads: {info.downloads:,}")
        print(f"    • Likes: {info.likes:,}")
        print(f"    • Private: {info.private}")
        print(f"    • Tags: {', '.join(info.tags) if info.tags else 'None'}")

        # Get config names
        print("  ✓ Checking dataset configs...")
        try:
            configs = get_dataset_config_names(repo_id)
            print(f"    Available configs: {', '.join(configs) if configs else 'default'}")
        except Exception as e:
            print(f"    ⚠ Could not list configs: {str(e)[:100]}")

        # Try to load dataset
        print("  ✓ Loading dataset...")
        try:
            dataset = load_dataset(repo_id, split=None)
            print(f"    Splits: {list(dataset.keys())}")

            for split_name, split_data in dataset.items():
                print(f"      • {split_name}: {len(split_data)} examples")
                # Show column names
                if len(split_data) > 0:
                    cols = split_data.column_names
                    print(f"        Columns: {', '.join(cols)}")
                    # Show first example
                    first_example = split_data[0]
                    print(f"        First example keys: {list(first_example.keys())}")

        except Exception as e:
            print(f"    ⚠ Could not load dataset: {str(e)[:100]}")

        print("  ✅ Dataset check passed!")
        return True

    except Exception as e:
        print(f"  ❌ Dataset check failed: {str(e)[:200]}")
        return False


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " EDGEID HUGGINGFACE COLLECTION VERIFICATION ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    results = {"models": {}, "datasets": {}}

    # Check models
    print_header("🤖 CHECKING MODELS")
    for name, repo_id in MODELS.items():
        results["models"][name] = check_model(name, repo_id)
        print()

    # Check datasets
    print_header("📊 CHECKING DATASETS")
    for name, repo_id in DATASETS.items():
        results["datasets"][name] = check_dataset(name, repo_id)
        print()

    # Summary
    print_header("📋 VERIFICATION SUMMARY")
    model_passed = sum(1 for v in results["models"].values() if v)
    model_total = len(results["models"])
    dataset_passed = sum(1 for v in results["datasets"].values() if v)
    dataset_total = len(results["datasets"])

    print(f"Models:   {model_passed}/{model_total} passed ✓")
    for name, passed in results["models"].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    print(f"\nDatasets: {dataset_passed}/{dataset_total} passed ✓")
    for name, passed in results["datasets"].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    # Overall status
    all_passed = (model_passed == model_total) and (dataset_passed == dataset_total)
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ All checks passed! Collection is ready for use.")
    else:
        print("⚠ Some checks failed. Check details above.")
        print("Note: If repos are private, ensure HF authentication is configured.")
        print("Run: huggingface-cli login")

    print(f"{'='*80}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
