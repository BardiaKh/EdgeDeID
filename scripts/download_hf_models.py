#!/usr/bin/env python3
"""
Download all EdgeDeID models from HuggingFace Hub into ./models/.

Usage:
    python scripts/download_hf_models.py              # all models
    python scripts/download_hf_models.py Qwen3-0.6B-GRPO-KL

The released repos are public; a token is only needed to push (huggingface-cli login, or set HF_TOKEN).
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "Qwen3-1.7B-Base_SFT": "EdgeDeID/Qwen3-1.7B-Base_SFT",
    "Qwen3-0.6B-Base_SFT": "EdgeDeID/Qwen3-0.6B-Base_SFT",
    "Qwen3-0.6B-GRPO-KL": "EdgeDeID/Qwen3-0.6B-GRPO-KL",
}

MODELS_DIR = Path(__file__).parent.parent / "models"


def download(name: str, repo_id: str) -> Path:
    target = MODELS_DIR / name
    print(f"Downloading {repo_id} -> {target}")
    path = snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"  done: {path}")
    return Path(path)


def main(argv: list[str]) -> int:
    names = argv or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(MODELS)}")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        try:
            download(name, MODELS[name])
        except Exception as e:
            print(f"  FAIL {name}: {str(e)[:200]}")
            failed.append(name)

    print(f"\n{len(names) - len(failed)}/{len(names)} model(s) downloaded to {MODELS_DIR}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
