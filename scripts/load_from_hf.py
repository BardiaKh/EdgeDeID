#!/usr/bin/env python3
"""
Load EdgeDeID synthetic data from Hugging Face.

Usage in notebooks:
    from scripts.load_from_hf import load_dataset, load_teachers
    
    # Load entire dataset
    df_deepseek = load_dataset("deepseek", pseudolabeled=True)
    
    # Load all teachers
    data = load_teachers(pseudolabeled=True)
    df_all = pd.concat([data["deepseek"], data["haiku"], data["hermes"]])

Token resolution (automatic):
  1. ~/.huggingface/token (local setup)
  2. HF_TOKEN environment variable
  3. .env file in repo root (Colab/Kaggle)

Caches files locally in ./data/ to avoid repeated downloads.
"""

import os
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


REPO_ID = "EdgeDeID/synthetic-dataset"
REPO_TYPE = "dataset"
CACHE_DIR = Path(__file__).parent.parent / "data"
TEACHERS = ["deepseek", "haiku", "hermes"]


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(teacher: str, pseudolabeled: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Load a single teacher's data from Hugging Face.
    
    Args:
        teacher: "deepseek", "haiku", or "hermes"
        pseudolabeled: If True, load {teacher}_pseudolabeled.csv; else load all output_N_*.csv
        verbose: Print download progress
    
    Returns:
        Pandas DataFrame with the loaded data
    
    Example:
        >>> df = load_dataset("haiku", pseudolabeled=True)
        >>> print(df.shape)  # (12345, N)
    """
    
    assert teacher in TEACHERS, f"Teacher must be one of {TEACHERS}"
    ensure_cache_dir()
    
    if pseudolabeled:
        filename = f"{teacher}/{teacher}_pseudolabeled.csv"
        if verbose:
            print(f"📥 Loading {teacher}_pseudolabeled.csv...")
        
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            cache_dir=str(CACHE_DIR.parent),
            local_dir=str(CACHE_DIR),
            local_dir_use_symlinks=False
        )
        
        df = pd.read_csv(file_path)
        if verbose:
            print(f"✓ Loaded {len(df)} rows from {teacher}")
        return df
    
    else:
        # Load all raw output_N_*.csv files
        if verbose:
            print(f"📥 Loading all raw outputs for {teacher}...")
        
        dfs = []
        i = 1
        while True:
            filename = f"{teacher}/output_{i}_{teacher}.csv"
            try:
                file_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type=REPO_TYPE,
                    cache_dir=str(CACHE_DIR.parent),
                    local_dir=str(CACHE_DIR),
                    local_dir_use_symlinks=False
                )
                df = pd.read_csv(file_path)
                dfs.append(df)
                if verbose:
                    print(f"  ✓ output_{i}_{teacher}.csv ({len(df)} rows)")
                i += 1
            except Exception:
                break  # No more files
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            if verbose:
                print(f"✓ Loaded {len(result)} total rows from {len(dfs)} files")
            return result
        else:
            raise FileNotFoundError(f"No data files found for {teacher}")


def load_teachers(pseudolabeled: bool = True, verbose: bool = True) -> dict:
    """
    Load data from all three teachers.
    
    Args:
        pseudolabeled: If True, load pseudolabeled versions
        verbose: Print progress
    
    Returns:
        Dictionary: {"deepseek": df, "haiku": df, "hermes": df}
    
    Example:
        >>> data = load_teachers(pseudolabeled=True)
        >>> df_combined = pd.concat(data.values(), ignore_index=True)
    """
    
    result = {}
    for teacher in TEACHERS:
        result[teacher] = load_dataset(teacher, pseudolabeled=pseudolabeled, verbose=verbose)
    
    return result


def download_all(verbose: bool = True):
    """
    Download the entire dataset to cache directory.
    
    One-time call to pre-populate all data locally.
    
    Example:
        >>> download_all()  # ~5-10 minutes first time
        >>> # Subsequent loads are instant
    """
    
    ensure_cache_dir()
    
    if verbose:
        print(f"📥 Downloading entire EdgeDeID dataset...")
        print(f"   Cache dir: {CACHE_DIR}")
    
    dataset_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        cache_dir=str(CACHE_DIR.parent),
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False
    )
    
    if verbose:
        print(f"✓ Dataset downloaded to {dataset_path}")
    
    return dataset_path


if __name__ == "__main__":
    # Quick test
    print("Testing data loading...")
    print("\n1. Loading haiku_pseudolabeled.csv")
    df_haiku = load_dataset("haiku", pseudolabeled=True)
    print(f"   Shape: {df_haiku.shape}")
    
    print("\n2. Loading all teachers")
    data = load_teachers(pseudolabeled=True)
    for teacher, df in data.items():
        print(f"   {teacher}: {df.shape[0]} rows")
