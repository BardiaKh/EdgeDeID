"""Build QC-verifier training data from the synthetic *training* split.

Every example pairs a training note with a candidate annotation and a verdict.
Candidates come from perturbing that note's gold annotation, plus (optionally)
real system outputs on the same training notes. MIMIC and the radiology
benchmark are never touched here — they are test sets.

Two rules keep the labels honest:

1. The label is always *recomputed* from the candidate with the zero-leak rule
   (``verifier.label_candidate``), never assumed from the perturbation. Dropping
   the variation "Chen" while "Wei Chen" survives still covers "Chen", so that
   "drop" is genuinely complete — and teaching it as incomplete would teach the
   verifier to flag harmless edits.
2. Every candidate is serialized canonically, and the cosmetic edits
   (reordering, dropping attributes) are applied to incomplete candidates as
   often as to complete ones. Neither formatting nor "was this edited at all"
   can then predict the label; only coverage can.

Perturbation families (weights in ``DEFAULT_OP_WEIGHTS``):

  gold           untouched gold                                   -> complete
  harmless       drop a name variant another variant contains      -> complete
  hallucinate    add a spurious entity (over-redaction)            -> complete
  drop_one       remove one entity or one name variation           -> usually incomplete
  drop_many      remove 2-4 units                                  -> incomplete
  truncate       shorten one span ("Amit Kumar" -> "Amit")         -> usually incomplete
  drop_category  remove a whole category (a system blind to it)    -> incomplete
  drop_and_add   drop real entities, add spurious ones             -> incomplete

``hallucinate`` matters: the verifier judges misses, not over-redaction, so it
must learn that extra entities do not make an annotation incomplete.
``drop_and_add`` keeps the entity count normal, so "fewer entities" cannot
stand in for actually checking coverage.

Shortcut audit (8k notes): note-blind features — entity count, empty
categories, candidate length, missing attributes — predict the label at
AUROC ~0.5-0.6, so a verifier must read the note to do well.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .verifier import (
    PARENTS,
    Annotation,
    Entity,
    build_single_pass_example,
    build_verifier_prompt,
    label_candidate,
    parse_annotation,
    serialize_annotation,
    verdict_completion,
)

DEFAULT_OP_WEIGHTS: dict[str, float] = {
    "gold": 0.10,
    "harmless": 0.12,
    "hallucinate": 0.10,
    "drop_one": 0.22,
    "drop_many": 0.12,
    "truncate": 0.14,
    "drop_category": 0.08,
    "drop_and_add": 0.12,
}

# Share of all candidates — every family, gold included — that also get
# cosmetic edits (reordering, stripped attributes). Applying it uniformly keeps
# "was reformatted" independent of the label.
COSMETIC_RATE = 0.5


# ─── perturbation primitives ──────────────────────────────────────────────────

def _cosmetic(ann: Annotation, rng: random.Random) -> Annotation:
    """Reorder entities within categories and drop id/type attributes."""
    a = ann.copy()
    for p in PARENTS:
        rng.shuffle(a.blocks[p])
        if rng.random() < 0.5:
            for e in a.blocks[p]:
                e.attrs = {}
    return a


def _drop_units(ann: Annotation, k: int, rng: random.Random) -> Annotation:
    """Remove k units. A unit is a whole entity or, for a name with several
    variations, sometimes just one variation."""
    a = ann.copy()
    for _ in range(k):
        units = a.units()
        if not units:
            break
        p, i = rng.choice(units)
        ent = a.blocks[p][i]
        if p == "names" and len(ent.values) > 1 and rng.random() < 0.5:
            ent.values.pop(rng.randrange(len(ent.values)))
        else:
            a.blocks[p].pop(i)
    return a


def _truncate_one(ann: Annotation, rng: random.Random) -> Annotation | None:
    """Shorten one span, leaving part of it uncovered."""
    a = ann.copy()
    choices = []
    for p in PARENTS:
        for i, e in enumerate(a.blocks[p]):
            # dates are matched on `original` (values[0]); names on each value
            idxs = [0] if p == "dates" else range(len(e.values))
            choices += [(p, i, j) for j in idxs if len(e.values[j]) >= 4]
    if not choices:
        return None
    p, i, j = rng.choice(choices)
    v = a.blocks[p][i].values[j]
    words = v.split()
    if len(words) >= 2:
        cut = rng.randint(1, len(words) - 1)
        v2 = " ".join(words[:cut])
    else:
        v2 = v[: max(1, int(len(v) * rng.uniform(0.4, 0.8)))]
    if v2 == v:
        return None
    a.blocks[p][i].values[j] = v2
    return a


def _drop_category(ann: Annotation, rng: random.Random) -> Annotation | None:
    a = ann.copy()
    nonempty = [p for p in PARENTS if a.blocks[p]]
    if not nonempty:
        return None
    a.blocks[rng.choice(nonempty)] = []
    return a


def _drop_contained_variant(ann: Annotation, rng: random.Random) -> Annotation:
    """Drop a name variation that another kept variation contains ("Chen" when
    "Wei Chen" stays). Coverage is unchanged; the label check confirms it."""
    a = ann.copy()
    cands = []
    for i, e in enumerate(a.blocks["names"]):
        low = [v.lower() for v in e.values]
        for j, v in enumerate(low):
            if any(k != j and v in w for k, w in enumerate(low)):
                cands.append((i, j))
    if cands:
        i, j = rng.choice(cands)
        a.blocks["names"][i].values.pop(j)
    return a


_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b|\b\d{2,}\b")


def _hallucinate(ann: Annotation, note: str, rng: random.Random) -> Annotation | None:
    """Add a spurious entity drawn from note text that no gold value contains."""
    a = ann.copy()
    gold_text = " ".join(v.lower() for p in PARENTS for e in a.blocks[p] for v in e.values)
    pool = [w for w in set(_WORD_RE.findall(note)) if w.lower() not in gold_text]
    if not pool:
        return None
    w = rng.choice(pool)
    # Give the spurious entity the same attributes real ones carry; a bare
    # element would mark this complete-class candidate as different.
    if w.isdigit():
        if rng.random() < 0.5:
            a.blocks["ages"].append(Entity({}, [w]))
        else:
            a.blocks["ids"].append(Entity({"type": "other"}, [w]))
    else:
        a.blocks["names"].append(Entity({"id": "0", "type": rng.choice(["patient", "provider", "other"])}, [w]))
    return a


def _drop_and_add(ann: Annotation, note: str, rng: random.Random) -> Annotation | None:
    """Drop real entities and add spurious ones: an incomplete candidate whose
    entity count looks normal, so "fewer entities" stops predicting the label."""
    n = len(ann.units())
    if n < 1:
        return None
    a = _drop_units(ann, rng.randint(1, min(2, n)), rng)
    for _ in range(rng.randint(1, 2)):
        a = _hallucinate(a, note, rng) or a
    return a


def perturb(op: str, gold: Annotation, note: str, rng: random.Random) -> Annotation | None:
    """Apply one perturbation family; None if it does not apply to this note."""
    if op == "gold":
        return gold.copy()
    if op == "harmless":
        return _drop_contained_variant(gold, rng)
    if op == "drop_and_add":
        return _drop_and_add(gold, note, rng)
    if op == "hallucinate":
        return _hallucinate(gold, note, rng)
    if op == "drop_one":
        return _drop_units(gold, 1, rng) if gold.units() else None
    if op == "drop_many":
        n = len(gold.units())
        return _drop_units(gold, rng.randint(2, 4), rng) if n >= 2 else None
    if op == "truncate":
        return _truncate_one(gold, rng)
    if op == "drop_category":
        return _drop_category(gold, rng)
    raise ValueError(f"unknown perturbation: {op}")


# ─── candidate generation ─────────────────────────────────────────────────────

@dataclass
class Candidate:
    note_idx: int
    note: str
    gold: str
    candidate: str
    source: str
    is_complete: bool


def synth_candidates(
    note_idx: int,
    note: str,
    gold_xml: str,
    n: int,
    rng: random.Random,
    op_weights: dict[str, float] = DEFAULT_OP_WEIGHTS,
) -> list[Candidate]:
    """``n`` perturbed candidates for one note, each with a recomputed label."""
    gold = parse_annotation(gold_xml)
    ops, w = zip(*op_weights.items())
    out: list[Candidate] = []
    tries = 0
    while len(out) < n and tries < 6 * n:
        tries += 1
        op = rng.choices(ops, weights=w)[0]
        ann = perturb(op, gold, note, rng)
        if ann is None:
            continue
        if rng.random() < COSMETIC_RATE:
            ann = _cosmetic(ann, rng)
        cand = serialize_annotation(ann)
        out.append(Candidate(note_idx, note, gold_xml, cand, op,
                             label_candidate(cand, gold_xml)))
    return out


def rebalance(cands: list[Candidate], incomplete_frac: float, rng: random.Random) -> list[Candidate]:
    """Downsample the majority class to hit ``incomplete_frac``; keep the rest."""
    inc = [c for c in cands if not c.is_complete]
    com = [c for c in cands if c.is_complete]
    if not inc or not com:
        return list(cands)
    # keep all of the scarcer side, trim the other
    want_com = int(round(len(inc) * (1 - incomplete_frac) / incomplete_frac))
    if want_com <= len(com):
        com = rng.sample(com, want_com)
    else:
        want_inc = int(round(len(com) * incomplete_frac / (1 - incomplete_frac)))
        inc = rng.sample(inc, min(want_inc, len(inc)))
    out = inc + com
    rng.shuffle(out)
    return out


def split_notes(n_total: int, n_notes: int, fracs: dict[str, float], seed: int) -> dict[str, list[int]]:
    """Disjoint note-index pools, so no note appears in two splits."""
    rng = random.Random(seed)
    idx = rng.sample(range(n_total), min(n_notes, n_total))
    out, start = {}, 0
    names = list(fracs)
    for k, name in enumerate(names):
        end = len(idx) if k == len(names) - 1 else start + int(round(fracs[name] * len(idx)))
        out[name] = sorted(idx[start:end])
        start = end
    return out


# ─── example assembly ─────────────────────────────────────────────────────────

def to_prefill_example(c: Candidate) -> dict[str, Any]:
    """Verification example: candidate in the prompt, verdict as completion."""
    return {
        "prompt": build_verifier_prompt(c.note, c.candidate),
        "completion": verdict_completion(c.is_complete),
        "is_complete": c.is_complete,
        "source": c.source,
        "note_idx": c.note_idx,
        "kind": "prefill",
    }


def to_grpo_example(c: Candidate) -> dict[str, Any]:
    """GRPO prompt: no completion — the reward reads ``is_complete``."""
    return {
        "prompt": build_verifier_prompt(c.note, c.candidate),
        "is_complete": c.is_complete,
        "source": c.source,
        "note_idx": c.note_idx,
    }


def to_single_pass_example(note_idx: int, note: str, gold_xml: str) -> dict[str, Any]:
    ex = build_single_pass_example(note, gold_xml)
    ex.update(source="single_pass", note_idx=note_idx, kind="single_pass")
    return ex


def load_extra_candidates(rows: Iterable[dict[str, Any]], gold_by_note: dict[str, int],
                          notes: list[str], golds: list[str]) -> list[Candidate]:
    """Real system outputs on *training* notes, joined to gold by note text.

    Rows need ``original`` (the training note) and ``candidate`` (the system's
    XML); ``source`` is optional. Rows whose note is not in the training split
    are dropped, which also guards against feeding test notes in by mistake.
    """
    out = []
    for r in rows:
        i = gold_by_note.get(str(r.get("original", "")).strip())
        if i is None:
            continue
        cand = serialize_annotation(parse_annotation(str(r.get("candidate", ""))))
        out.append(Candidate(i, notes[i], golds[i], cand,
                             f"model:{r.get('source', 'extra')}",
                             label_candidate(cand, golds[i])))
    return out
