#!/usr/bin/env python3
"""
Load EdgeDeID models (SFT / GRPO) from the Hugging Face collection.

Usage in notebooks / scripts:
    from scripts.load_model_from_hf import load_model

    # By short name (see MODELS registry below)
    model, tokenizer = load_model("grpo")

    # Or by full repo id
    model, tokenizer = load_model("EdgeDeID/Qwen3-0.6B-GRPO")

    # Pre-download weights without instantiating the model
    from scripts.load_model_from_hf import download_model
    download_model("grpo")

CLI:
    python scripts/load_model_from_hf.py            # loads the GRPO model
    python scripts/load_model_from_hf.py sft-0.6b   # loads an SFT model

Token resolution (automatic, models are private/gated):
  1. ~/.huggingface/token (local setup)
  2. .env file in repo root (Colab/Kaggle)
  3. HF_TOKEN environment variable
  4. None (huggingface_hub default)

Caches weights locally under ./models/ to avoid repeated downloads.

Requires: huggingface_hub, transformers, torch
"""

import os
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Short name -> HF repo id. Keep in sync with scripts/verify_hf_collection.py.
MODELS = {
    "grpo": "EdgeDeID/Qwen3-0.6B-GRPO",
    "sft-0.6b": "EdgeDeID/Qwen3-0.6B-Base_SFT",
    "sft-1.7b": "EdgeDeID/Qwen3-1.7B-Base_SFT",
}

DEFAULT_MODEL = "grpo"
CACHE_DIR = Path(__file__).parent.parent / "models"


def load_token():
    """Resolve an HF token from (in order): ~/.huggingface/token, .env, HF_TOKEN env var."""
    hf_token_path = Path.home() / ".huggingface" / "token"
    if hf_token_path.exists():
        print(f"✓ Using token from {hf_token_path}")
        return hf_token_path.read_text().strip()

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        match = re.search(r"HF_TOKEN\s*=\s*(.+)", env_path.read_text())
        if match:
            print("✓ Using HF_TOKEN from .env")
            return match.group(1).strip().strip('"').strip("'")

    if "HF_TOKEN" in os.environ:
        print("✓ Using HF_TOKEN from environment variable")
        return os.environ["HF_TOKEN"]

    print("ℹ️  No explicit token found. Using default HuggingFace token...")
    return None


def resolve_repo_id(name: str) -> str:
    """Accept either a short name from MODELS or a full 'owner/repo' id."""
    if name in MODELS:
        return MODELS[name]
    if "/" in name:
        return name
    raise ValueError(f"Unknown model '{name}'. Use a full repo id or one of: {list(MODELS)}")


def download_model(name: str = DEFAULT_MODEL, token: str | None = None, verbose: bool = True) -> str:
    """
    Download all model files to ./models/<repo-name>/ and return the local path.

    Does not instantiate the model — useful for pre-populating the cache offline.
    """
    repo_id = resolve_repo_id(name)
    if token is None:
        token = load_token()

    local_dir = CACHE_DIR / repo_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📥 Downloading {repo_id} -> {local_dir} ...")

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    if verbose:
        print(f"✓ Downloaded to {path}")
    return path


def load_model(
    name: str = DEFAULT_MODEL,
    token: str | None = None,
    device_map: str | None = None,
    dtype: str = "auto",
    verbose: bool = True,
):
    """
    Download (if needed) and load an EdgeDeID model + tokenizer from HF.

    Args:
        name:       short name from MODELS ("grpo", "sft-0.6b", ...) or a full repo id.
        token:      HF token; auto-resolved if None.
        device_map: passed to from_pretrained (e.g. "cuda:0", "auto"); defaults to
                    "cuda:0" when a GPU is available, else "cpu".
        dtype:      torch dtype for the model ("auto", "float16", "bfloat16", "float32").
        verbose:    print progress.

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_id = resolve_repo_id(name)
    if token is None:
        token = load_token()

    # Cache weights locally, then load from disk for reproducible offline use.
    local_path = download_model(name, token=token, verbose=verbose)

    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)

    if verbose:
        print(f"🧠 Loading {repo_id} (device_map={device_map}, dtype={dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(local_path, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch_dtype,
        device_map=device_map,
        token=token,
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Loaded {repo_id}: {n_params/1e6:.1f}M params, tokenizer vocab={len(tokenizer)}")

    return model, tokenizer


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"🚀 Loading EdgeDeID model: {target}")
    print("=" * 60)
    model, tokenizer = load_model(target)
    print("\nDone.")
