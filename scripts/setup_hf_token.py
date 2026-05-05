#!/usr/bin/env python3
"""
Setup HuggingFace token securely for EdgeDeID team collaboration.

Usage:
    python scripts/setup_hf_token.py

This guide shows TWO secure ways to store HF tokens:
1. Local machine (recommended): ~/.huggingface/token
2. Colab/Kaggle (shared notebooks): .env file (gitignored)
"""

import os
import sys
from pathlib import Path


def setup_local():
    """Setup HF token locally (for local development)."""
    print("\n" + "=" * 70)
    print("🔐 LOCAL SETUP (Recommended for local development)")
    print("=" * 70)
    print("""
Each team member generates a PERSONAL HF token:

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "EdgeDeID team" or your name
4. Permissions: Read + Write access
5. Copy the token

Then run this command once:
    huggingface-cli login

Paste your token when prompted. It's saved to:
    ~/.huggingface/token

✓ That's it! No files in the repo, no copying needed.
✓ Scripts automatically use your token.
✓ Everyone has their own token (personal accountability).
""")

    # Check if token already exists
    token_path = Path.home() / ".huggingface" / "token"
    if token_path.exists():
        print(f"✓ Token found at: {token_path}")
        return True
    else:
        print(f"⚠️  No token found at: {token_path}")
        print("   Run: huggingface-cli login")
        return False


def setup_colab():
    """Setup HF token for Colab/Kaggle (fallback)."""
    print("\n" + "=" * 70)
    print("☁️  COLAB/KAGGLE SETUP (Shared notebooks)")
    print("=" * 70)
    print("""
For shared Colab/Kaggle notebooks:

1. Create .env file in EdgeDeID repo root:
    cp .env.example .env

2. Edit .env and replace:
    HF_TOKEN=your_token_here

3. Never commit .env (it's in .gitignore):
    cat .gitignore | grep ".env"  # Verify it's ignored

✓ .env is gitignored (safe)
✓ Scripts read HF_TOKEN from .env
✓ Upload notebook uses your token automatically
""")

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"✓ .env file exists at: {env_path}")
        return True
    else:
        print(f"⚠️  No .env file found")
        print(f"   Create one: cp .env.example .env")
        return False


def verify_setup():
    """Verify that a token is accessible."""
    print("\n" + "=" * 70)
    print("✅ VERIFICATION")
    print("=" * 70)

    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ HF token is valid!")
        print(f"  User: {user_info['name']}")
        print(f"  Access: {user_info['type']}")
        return True
    except Exception as e:
        print(f"❌ Token not found or invalid: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: huggingface-cli login")
        print("  2. Or create .env file with HF_TOKEN")
        return False


def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║            EdgeDeID HuggingFace Token Setup                        ║
║         (Secure, Team-friendly, Zero Copy-Pasting)                ║
╚════════════════════════════════════════════════════════════════════╝
""")

    # Show options
    setup_local()
    setup_colab()

    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS")
    print("=" * 70)
    print("""
For LOCAL development:
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token (name: "EdgeDeID")
  3. Run: huggingface-cli login
  4. Paste your token
  5. Verify: python scripts/setup_hf_token.py --verify

For COLAB/KAGGLE:
  1. Copy .env.example to .env
  2. Add your HF token to .env
  3. Run upload notebook (scripts automatically read .env)

Then push data:
  python scripts/push_to_hf.py
""")

    # Optionally verify
    if "--verify" in sys.argv:
        verify_setup()


if __name__ == "__main__":
    main()
