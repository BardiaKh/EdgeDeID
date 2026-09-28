"""Tests for the QC verifier (edgedeid.qc.verifier / verifier_data / verifier_eval).

Run:  python -m pytest tests/test_qc_verifier.py -q

Fast tests need nothing. The scoring-equivalence test builds a tiny random-init
Qwen3 and needs the EdgeDeID tokenizer (HF cache or HF_TOKEN); it is skipped
otherwise. No test downloads model weights.
"""
from __future__ import annotations

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgedeid.eval.metrics import compute_metrics, note_is_complete  # noqa: E402
from edgedeid.qc.verifier import (  # noqa: E402
    QC_OPEN,
    build_verifier_prompt,
    canonicalize,
    label_candidate,
    parse_annotation,
    parse_verdict,
    serialize_annotation,
    verifier_reward_one,
)
from edgedeid.qc.verifier_data import perturb, rebalance, split_notes, synth_candidates  # noqa: E402
from edgedeid.qc.verifier_eval import auroc, operating_point  # noqa: E402

GOLD = """<phi_entities>
<names>
<name id="1" type="patient">
<variations>
<value>Wei Chen</value>
<value>Chen</value>
</variations>
</name>
<name id="2" type="provider">
<variations>
<value>Rebecca Goldstein</value>
</variations>
</name>
</names>
<ages>
<age>37</age>
</ages>
<dates>
<date>
<original>March 3, 2024</original>
<formatted>2024-03-03</formatted>
</date>
</dates>
<places>
<place id="1" type="hospital">Mount Sinai Hospital</place>
</places>
<addresses/>
<phones>
<phone>(917) 555-0198</phone>
</phones>
<emails/>
<ids>
<id type="mrn">MSH-56789012</id>
</ids>
<urls/>
<devices/>
</phi_entities>"""

NOTE = ("Wei Chen, a 37-year-old, was seen at Mount Sinai Hospital on March 3, 2024 "
        "by Dr. Rebecca Goldstein. Chen can be reached at (917) 555-0198. "
        "MRN MSH-56789012. Follow-up with Cardiology in Boston.")


# ─── label definition ─────────────────────────────────────────────────────────

def test_canonical_round_trip_is_byte_exact():
    assert canonicalize(GOLD) == GOLD


def test_note_is_complete_matches_zero_leak_rate():
    preds = [GOLD, canonicalize(GOLD).replace("<value>Wei Chen</value>\n", ""), "garbage"]
    golds = [GOLD] * 3
    zl = compute_metrics(preds, golds).zero_leak_rate
    assert zl == sum(note_is_complete(p, g) for p, g in zip(preds, golds)) / 3


def test_partial_span_is_a_leak_unlike_entity_recall():
    # "Wei" does not cover "Wei Chen": the metric counts the uncovered part as
    # leaked. (rl.rewards.entity_recall would call this caught.)
    ann = parse_annotation(GOLD)
    ann.blocks["names"][0].values = ["Wei"]
    assert not label_candidate(serialize_annotation(ann), GOLD)


def test_dropping_a_contained_variation_stays_complete():
    ann = parse_annotation(GOLD)
    ann.blocks["names"][0].values = ["Wei Chen"]          # "Chen" still covered
    assert label_candidate(serialize_annotation(ann), GOLD)


def test_extra_entities_do_not_make_incomplete():
    ann = perturb("hallucinate", parse_annotation(GOLD), NOTE, random.Random(0))
    assert ann is not None
    assert len(ann.units()) > len(parse_annotation(GOLD).units())
    assert label_candidate(serialize_annotation(ann), GOLD)


def test_unparseable_candidate_is_incomplete_and_canonically_empty():
    assert not label_candidate("<phi_entities><names>", GOLD)
    assert parse_annotation("<phi_entities><names>").units() == []


# ─── shortcut guards ──────────────────────────────────────────────────────────

def test_ids_renumbered_so_drops_leave_no_gap():
    ann = parse_annotation(GOLD)
    ann.blocks["names"].pop(0)                              # drop name id=1
    out = serialize_annotation(ann)
    assert 'id="1" type="provider"' in out and 'id="2"' not in out


def test_every_perturbation_is_canonical_and_relabelled():
    rng = random.Random(1)
    cands = synth_candidates(0, NOTE, GOLD, 60, rng)
    assert cands
    for c in cands:
        assert canonicalize(c.candidate) == c.candidate   # one serialization for all
        assert c.is_complete == label_candidate(c.candidate, GOLD)
    assert {c.is_complete for c in cands} == {True, False}


def test_rebalance_hits_target_fraction():
    rng = random.Random(2)
    cands = []
    for i in range(40):
        cands += synth_candidates(i, NOTE, GOLD, 5, rng)
    out = rebalance(cands, 0.3, rng)
    frac = sum(not c.is_complete for c in out) / len(out)
    assert abs(frac - 0.3) < 0.02


def test_note_pools_are_disjoint():
    pools = split_notes(1000, 800, {"sft": 0.7, "grpo": 0.2, "val": 0.1}, seed=3)
    a, b, c = (set(pools[k]) for k in ("sft", "grpo", "val"))
    assert not (a & b or a & c or b & c) and len(a | b | c) == 800


# ─── prompt / verdict / reward ────────────────────────────────────────────────

def test_prompt_ends_at_open_tag_with_candidate_last():
    p = build_verifier_prompt(NOTE, GOLD)
    assert p.endswith(f"</phi_entities>\n{QC_OPEN}")
    assert p.rsplit("</note>\n\n", 1)[1].startswith("<phi_entities>")


@pytest.mark.parametrize("text,want", [
    ("complete</qc_status>", "complete"),
    (" incomplete</qc_status>", "incomplete"),
    ("xx<qc_status>incomplete</qc_status>", "incomplete"),
    ("maybe", None),
])
def test_parse_verdict(text, want):
    assert parse_verdict(text) == want


def test_reward_ordering():
    correct = verifier_reward_one("incomplete</qc_status>", is_complete=False)
    cautious = verifier_reward_one("incomplete</qc_status>", is_complete=True)
    overconf = verifier_reward_one("complete</qc_status>", is_complete=False)
    malformed = verifier_reward_one("I think so", is_complete=False)
    assert correct == 1.0 and malformed == 0.0
    assert correct > cautious > overconf > malformed


def test_reward_implied_threshold_is_one_third():
    # expected reward of each verdict when P(incomplete)=p
    def e(v, p):
        return p * verifier_reward_one(v, False) + (1 - p) * verifier_reward_one(v, True)
    for p in (0.30, 0.36):
        says_inc = e("incomplete</qc_status>", p) > e("complete</qc_status>", p)
        assert says_inc == (p > 1 / 3)


# ─── evaluation metrics ───────────────────────────────────────────────────────

def test_operating_point_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    p = [.9, .8, .2, .7, .1, .1, .1, .1, .1, .1]
    op = operating_point(y, p, 0.5)
    assert op["flag_rate"] == 0.3
    assert math.isclose(op["catch_rate"], 2 / 3)
    assert math.isclose(op["residual_leak_rate"], 1 / 7)
    # positives .9,.8,.2 vs negatives .7 + six .1: 7 + 7 + 6 winning pairs of 21
    assert auroc(y, p) == pytest.approx(20 / 21)


# ─── scoring equivalence (tiny random model, real tokenizer) ──────────────────

def _tokenizer_or_skip():
    try:
        from edgedeid.qc.train_utils import load_env
        load_env(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("EdgeDeID/Qwen3-0.6B-Base_SFT",
                                             token=os.environ.get("HF_TOKEN"))
    except Exception as e:                                 # no cache, no token, offline
        pytest.skip(f"EdgeDeID tokenizer unavailable: {e}")


def test_batched_scoring_equals_brute_force():
    """Left-padded, length-sorted, tail-logit batched scoring must equal the
    unbatched full-sequence log-probabilities."""
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from edgedeid.qc.verifier import score_incomplete, verdict_token_ids

    tok = _tokenizer_or_skip()
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=8192)
    model = Qwen3ForCausalLM(cfg).eval()

    rng = random.Random(4)
    prompts = []
    for k in range(7):                                    # varied lengths -> padding
        ann = parse_annotation(GOLD)
        if k % 2:
            ann.blocks["names"].pop(0)
        prompts.append(build_verifier_prompt(NOTE * (1 + k % 3), serialize_annotation(ann)))
    rng.shuffle(prompts)

    got = score_incomplete(model, tok, prompts, batch_size=3, progress=False)

    vt = verdict_token_ids(tok)

    def full_lp(prompt, cont):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids + cont])
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=x).logits.float(), -1)[0]
        return sum(lp[len(ids) - 1 + k, cont[k]].item() for k in range(len(cont)))

    for p, g in zip(prompts, got):
        li, lc = full_lp(p, vt.incomplete), full_lp(p, vt.complete)
        want = 1 / (1 + math.exp(lc - li))
        assert g == pytest.approx(want, abs=1e-4)
