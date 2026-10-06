"""GRPO must train with a KL anchor to the SFT reference.

Before 2026-10-02 the project's GRPOConfig had no ``beta`` and build_grpo_trainer
never set one, so TRL's default beta=0.0 applied to every run. Unanchored, the
policy drifted wherever the reward was silent: the published GRPO checkpoint
writes the wrong ISO year in every <formatted> date.
"""
import warnings

import pytest

from edgedeid.rl.config import GRPOConfig


def _cfg(tmp_path, **kw):
    base = dict(model_name="x", output_dir=str(tmp_path), max_steps=2, num_generations=2,
                per_device_train_batch_size=2, gradient_accumulation_steps=1,
                max_prompt_length=None, max_completion_length=8, logging_steps=1)
    # (logging_steps must be >0; GRPOConfig default is 5)
    base.update(kw)
    return GRPOConfig(**base)


def test_default_beta_is_nonzero():
    assert GRPOConfig(model_name="x", output_dir="y").beta > 0


def test_beta_reaches_trl(tmp_path, monkeypatch):
    import trl.trainer.grpo_trainer as gt
    from edgedeid.rl import grpo

    seen = {}

    class FakeTrainer:
        def __init__(self, *, args, **kw):
            seen["beta"] = args.beta

    monkeypatch.setattr(gt, "GRPOTrainer", FakeTrainer)
    grpo.build_grpo_trainer(None, None, _cfg(tmp_path, beta=0.07), lambda c, **k: [0.0] * len(c), [])
    assert seen["beta"] == 0.07


def test_beta_zero_warns(tmp_path, monkeypatch):
    import trl.trainer.grpo_trainer as gt
    from edgedeid.rl import grpo

    monkeypatch.setattr(gt, "GRPOTrainer", lambda **kw: None)
    with pytest.warns(UserWarning, match="no KL anchor"):
        grpo.build_grpo_trainer(None, None, _cfg(tmp_path, beta=0.0), lambda c, **k: [0.0] * len(c), [])


def test_kl_is_active_in_a_real_step(tmp_path):
    """Two real GRPO steps on a tiny random Qwen3: a reference model exists and KL is logged."""
    torch = pytest.importorskip("torch")
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from edgedeid.rl.grpo import build_grpo_trainer

    try:
        tok = AutoTokenizer.from_pretrained("EdgeDeID/Qwen3-0.6B-Base_SFT")
    except Exception as e:  # no network / no HF token
        pytest.skip(f"tokenizer unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                        head_dim=16, max_position_embeddings=256, tie_word_embeddings=True))
    # With beta > 0 TRL rebuilds the reference from model.config._name_or_path, as it
    # does from the hub id in real training -- so the tiny model must live on disk.
    tiny.save_pretrained(tmp_path / "tiny")
    model = Qwen3ForCausalLM.from_pretrained(tmp_path / "tiny")
    ds = Dataset.from_list([{"prompt": f"<note>\nnote {i}\n</note>\n\n"} for i in range(4)])
    reward = lambda completions, **kw: [float(len(c) % 3) for c in completions]  # any non-constant signal
    trainer = build_grpo_trainer(model, tok, _cfg(tmp_path / "out", beta=0.04), reward, ds)
    assert trainer.beta == 0.04 and trainer.ref_model is not None
    trainer.train()
    kl = [h["kl"] for h in trainer.state.log_history if "kl" in h]
    assert kl, "TRL logged no KL -- the KL term is not active"
