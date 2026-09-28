"""Shared helpers for the QC-verifier training scripts.

Kept small and version-tolerant: pyproject pins transformers<5 / trl>=0.23, but
the scripts also have to run on trl 1.x. Config dataclasses differ between
those versions (trl 1.x dropped ``max_prompt_length``; trl 0.23 still has it
with a 512-token default that LEFT-truncates prompts, silently cutting the
instruction block), so configs are built by filtering against the installed
class's fields rather than by hard-coding keyword arguments.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from .verifier_eval import auroc, operating_point


def load_env(path: str = ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (HF_TOKEN for gated repos)."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_config(cls, **kwargs):
    """Instantiate a (TRL) config dataclass, dropping keys this version lacks.

    Dropped keys are printed, so a silently ignored setting is visible in logs.
    """
    fields = {f.name for f in dataclasses.fields(cls)}
    kept = {k: v for k, v in kwargs.items() if k in fields}
    dropped = sorted(set(kwargs) - fields)
    if dropped:
        print(f"[{cls.__name__}] not supported by this TRL version, ignored: {dropped}")
    return cls(**kept)


def load_base(model_id: str, *, dtype: str = "bfloat16", attn: str | None = None,
              token: str | None = None):
    """Load the extraction SFT checkpoint and its tokenizer for adapter training.

    The checkpoint's config.json was saved with ``use_cache=False`` (inherited
    from gradient-checkpointed SFT). Training keeps it off; generation code sets
    ``use_cache=True`` explicitly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict[str, Any] = {"dtype": getattr(torch, dtype), "token": token}
    if attn:
        kw["attn_implementation"] = attn
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except TypeError:                       # transformers < 4.56 uses torch_dtype
        kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    # The tokenizer must already carry the schema tokens the SFT run added.
    emb = model.get_input_embeddings().weight.shape[0]
    if len(tok) > emb:
        raise ValueError(f"tokenizer has {len(tok)} tokens but model embeddings {emb}: "
                         "load the tokenizer saved with the SFT checkpoint")
    return model, tok


def val_auroc(model, tok, val_rows: list[dict], *, batch_size: int = 4,
              limit: int | None = 600) -> dict[str, float]:
    """Quick in-distribution check on synthetic validation rows."""
    from .verifier import score_incomplete

    rows = [r for r in val_rows if r.get("kind", "prefill") == "prefill"][:limit]
    if not rows:
        return {}
    was_training = model.training
    model.eval()
    p = score_incomplete(model, tok, [r["prompt"] for r in rows], batch_size=batch_size)
    if was_training:
        model.train()
    y = [0 if r["is_complete"] else 1 for r in rows]
    op = operating_point(y, p, 1 / 3)                     # reward-implied threshold
    return {
        "val_n": len(rows),
        "val_incomplete_frac": sum(y) / len(y),
        "val_auroc": auroc(y, p),
        "val_catch_rate@1/3": op["catch_rate"],
        "val_flag_rate@1/3": op["flag_rate"],
    }
