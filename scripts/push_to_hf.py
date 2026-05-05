#!/usr/bin/env python3
"""
Push EdgeDeID synthetic data to Hugging Face dataset repo.

Usage:
    python scripts/push_to_hf.py

Uploads data directly from ./data to HF
Token resolution (automatic):
  1. ~/.huggingface/token (local setup)
  2. .env file (Colab/Kaggle)
  3. HF_TOKEN environment variable

Requires: huggingface_hub, git-lfs, valid HF token
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from tqdm import tqdm

# Configuration
REPO_ID = "EdgeDeID/synthetic-dataset"
REPO_TYPE = "dataset"
DATA_DIR = Path(__file__).parent.parent / "data"
TEACHERS = ["deepseek", "haiku", "hermes"]


def load_token():
    """
    Load HF token from multiple sources (in order of precedence):
    1. ~/.huggingface/token (local setup)
    2. .env file in repo root (Colab/Kaggle)
    3. HF_TOKEN environment variable
    4. None (will use huggingface_hub default)
    """
    # Check ~/.huggingface/token
    hf_token_path = Path.home() / ".huggingface" / "token"
    if hf_token_path.exists():
        token = hf_token_path.read_text().strip()
        print(f"✓ Using token from {hf_token_path}")
        return token
    
    # Check .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            import re
            env_content = env_path.read_text()
            match = re.search(r'HF_TOKEN\s*=\s*(.+)', env_content)
            if match:
                token = match.group(1).strip()
                print(f"✓ Using HF_TOKEN from .env")
                return token
        except Exception:
            pass
    
    # Check environment variable
    if "HF_TOKEN" in os.environ:
        print("✓ Using HF_TOKEN from environment variable")
        return os.environ["HF_TOKEN"]
    
    # Let huggingface_hub use its default
    print("ℹ️  No explicit token found. Using default HuggingFace token...")
    return None


def get_files_to_upload():
    """Collect all CSV files from /data/{teacher}/ directories."""
    files = []
    
    for teacher in TEACHERS:
        teacher_dir = DATA_DIR / teacher
        if not teacher_dir.exists():
            print(f"⚠️  {teacher_dir} not found, skipping...")
            continue
        
        csv_files = list(teacher_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} files in {teacher}/")
        
        for csv_file in csv_files:
            # HF path: no /data/ prefix, just deepseek/file.csv
            hf_path = f"{teacher}/{csv_file.name}"
            files.append((csv_file, hf_path))
    
    return files


def push_data_to_hf(token=None):
    """Push all CSV files to HF dataset repo."""
    
    api = HfApi(token=token)
    
    try:
        # Verify repo exists and we have access
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"✓ Connected to {REPO_ID}")
        print(f"  Private: {repo_info.private}")
    except Exception as e:
        print(f"❌ Error: Cannot access {REPO_ID}")
        print(f"   Make sure: 1) Repo exists  2) You have write access  3) Token is valid")
        print(f"   Error: {e}")
        sys.exit(1)
    
    # Collect files
    files_to_upload = get_files_to_upload()
    if not files_to_upload:
        print("❌ No CSV files found in /data/")
        sys.exit(1)
    
    print(f"\n📤 Uploading {len(files_to_upload)} files...")
    print("=" * 60)
    
    # Create commit operations
    operations = []
    total_size = 0
    
    for local_path, hf_path in files_to_upload:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        total_size += local_path.stat().st_size
        print(f"  {hf_path:<45} ({size_mb:>6.1f} MB)")
        
        operations.append(
            CommitOperationAdd(
                path_in_repo=hf_path,
                path_or_fileobj=str(local_path)
            )
        )
    
    print("=" * 60)
    print(f"Total: {len(operations)} files, {total_size / (1024**2):.1f} MB")
    
    # Upload with a single commit
    try:
        print("\n⏳ Uploading to Hugging Face (this may take few minutes)...")
        
        commit_info = api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=operations,
            commit_message="Push all synthetic data (deepseek, haiku, hermes) - raw + pseudolabeled"
        )
        
        print(f"\n✓ Successfully uploaded!")
        print(f"  Commit: {commit_info.commit_url}")
        print(f"  View dataset: https://huggingface.co/datasets/{REPO_ID}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check your HF token: huggingface-cli login")
        print("  - Check internet connection")
        print("  - Check repo permissions")
        sys.exit(1)


if __name__ == "__main__":
    print("🚀 EdgeDeID → Hugging Face Dataset Push")
    print("=" * 60)
    
    # Load token automatically
    token = load_token()
    
    push_data_to_hf(token=token)
