"""Learned QC verifier: decide whether a <phi_entities> annotation is complete.

Why this exists
---------------
The first design asked the de-identifier to append ``<qc_status>`` to its own
output and trained that tag with GRPO. It never emitted the tag. The SFT targets
end at ``</phi_entities>``, so the policy put ~zero probability on anything
after it; every sampled completion in a GRPO group then scored 0 on the tag
term, the term had zero within-group variance, and it contributed no gradient.
GRPO can sharpen a behaviour the policy already produces — it cannot create one.

The verifier fixes that by construction. The candidate annotation is placed in
the *prompt* and the model generates only the verdict:

    <instructions>…</instructions>          (the same block SFT/GRPO used)
    <note>…</note>
    <phi_entities>…candidate…</phi_entities>
    <qc_status>                              <- prompt ends here
    complete</qc_status> | incomplete</qc_status>   <- completion

Consequences:

* Loss (SFT) and reward (GRPO) touch only the verdict, so a deliberately
  corrupted candidate can never teach the model to *omit* entities.
* The output is one of two strings, so both verdicts are in-support from the
  first step and GRPO groups have variance immediately.
* The candidate can come from any system — our model, Presidio, OPF, a human —
  which is what makes this usable as a system-agnostic review filter.
* The same format also supports single-pass use: prompt ends at ``</note>``,
  the model writes the XML itself, and the verdict follows (tag at the end).

Every candidate, at training and at evaluation time, goes through
``canonicalize`` first. Otherwise a verifier could learn a formatting shortcut
(corrupted candidates re-serialized one way, untouched gold another) instead of
checking coverage. Canonical form reproduces the gold training format exactly.

Label definition: ``edgedeid.eval.metrics.note_is_complete`` — the same rule as
the reported zero-leak metric. Not ``rl.rewards.entity_recall``, which treats a
partial span ("Amit" for "Amit Kumar") as caught.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Sequence
from xml.sax.saxutils import escape

from ..eval.metrics import note_is_complete
from ..rl.data import GRPO_SYSTEM_PROMPT, format_preference_prompt

QC_OPEN = "<qc_status>"
QC_CLOSE = "</qc_status>"
COMPLETE = "complete"
INCOMPLETE = "incomplete"

# Fixed order and singular forms of the ten category blocks, as in the gold.
PARENTS: tuple[str, ...] = (
    "names", "ages", "dates", "places", "addresses",
    "phones", "emails", "ids", "urls", "devices",
)
SINGULAR: dict[str, str] = {
    "names": "name", "ages": "age", "dates": "date", "places": "place",
    "addresses": "address", "phones": "phone", "emails": "email",
    "ids": "id", "urls": "url", "devices": "device",
}

_BLOCK_RE = re.compile(r"<phi_entities\b[^>]*>.*?</phi_entities>", re.DOTALL | re.IGNORECASE)
_VERDICT_RE = re.compile(r"^\s*(complete|incomplete)\s*</qc_status>", re.IGNORECASE)
_ANY_VERDICT_RE = re.compile(r"(?:<qc_status>\s*)?\b(complete|incomplete)\b", re.IGNORECASE)


# ─── structured candidate ─────────────────────────────────────────────────────

@dataclass
class Entity:
    """One entity element. ``values`` holds name variations, the date's
    (original, formatted) pair, or the single text value of anything else."""

    attrs: dict[str, str] = field(default_factory=dict)
    values: list[str] = field(default_factory=list)


@dataclass
class Annotation:
    """A parsed ``<phi_entities>`` block: category -> ordered entities."""

    blocks: dict[str, list[Entity]] = field(
        default_factory=lambda: {p: [] for p in PARENTS})

    def units(self) -> list[tuple[str, int]]:
        """(category, index) for every entity, in document order."""
        return [(p, i) for p in PARENTS for i in range(len(self.blocks[p]))]

    def copy(self) -> "Annotation":
        return Annotation({p: [Entity(dict(e.attrs), list(e.values)) for e in es]
                           for p, es in self.blocks.items()})


def parse_annotation(text: str | None) -> Annotation:
    """Parse any ``<phi_entities>`` block into an Annotation.

    Unparseable input yields an *empty* annotation. That matches the metric,
    where ``extract_entities`` of malformed XML is also empty, so the label and
    what the verifier sees stay consistent. (No repair here on purpose: repair
    would show the verifier entities the metric never credits.)
    """
    ann = Annotation()
    if not isinstance(text, str):
        return ann
    m = _BLOCK_RE.search(text)
    if not m:
        return ann
    try:
        root = ET.fromstring(m.group())
    except ET.ParseError:
        return ann
    for parent in PARENTS:
        pe = root.find(parent)
        if pe is None:
            continue
        for child in pe.findall(SINGULAR[parent]):
            attrs = dict(child.attrib)
            if parent == "names":
                vals = [v.text.strip() for v in child.iter("value")
                        if v.text and v.text.strip()]
                if vals:
                    ann.blocks[parent].append(Entity(attrs, vals))
            elif parent == "dates":
                orig = child.find("original")
                fmt = child.find("formatted")
                o = orig.text.strip() if orig is not None and orig.text else ""
                f = fmt.text.strip() if fmt is not None and fmt.text else ""
                if o or f:
                    ann.blocks[parent].append(Entity(attrs, [o, f]))
            elif child.text and child.text.strip():
                ann.blocks[parent].append(Entity(attrs, [child.text.strip()]))
    return ann


def _esc(text: str) -> str:
    """Escape as the gold does: &, <, > and double quotes (apostrophes stay)."""
    return escape(str(text), {'"': "&quot;"})


def _open(tag: str, attrs: dict[str, str]) -> str:
    a = "".join(f' {k}="{_esc(v)}"' for k, v in attrs.items())
    return f"<{tag}{a}>"


def serialize_annotation(ann: Annotation) -> str:
    """Emit the exact gold training format: one element per line, no
    indentation, all ten category blocks in fixed order, empty ones self-closed.

    Name and place ``id`` attributes are renumbered 1..n in document order.
    Gold ids are already sequential, so this is a no-op there; but after an
    entity is dropped the survivors would read "1, 3" — a gap that only
    corrupted candidates have, which a verifier could learn as a shortcut for
    "incomplete". The metric ignores ids, so renumbering loses nothing.
    """
    out = ["<phi_entities>"]
    for parent in PARENTS:
        ents = ann.blocks[parent]
        if not ents:
            out.append(f"<{parent}/>")
            continue
        out.append(f"<{parent}>")
        tag = SINGULAR[parent]
        next_id = 1
        for e in ents:
            if "id" in e.attrs and parent in ("names", "places"):
                e = Entity(dict(e.attrs, id=str(next_id)), e.values)
                next_id += 1
            if parent == "names":
                out += [_open(tag, e.attrs), "<variations>"]
                out += [f"<value>{_esc(v)}</value>" for v in e.values]
                out += ["</variations>", f"</{tag}>"]
            elif parent == "dates":
                o, f = (e.values + ["", ""])[:2]
                out += [_open(tag, e.attrs),
                        f"<original>{_esc(o)}</original>",
                        f"<formatted>{_esc(f)}</formatted>",
                        f"</{tag}>"]
            else:
                out.append(f"{_open(tag, e.attrs)}{_esc(e.values[0])}</{tag}>")
        out.append(f"</{parent}>")
    out.append("</phi_entities>")
    return "\n".join(out)


def canonicalize(candidate_xml: str | None) -> str:
    """Normalize any system's output to the canonical gold format."""
    return serialize_annotation(parse_annotation(candidate_xml))


# ─── prompts and targets ──────────────────────────────────────────────────────

def extraction_prefix(note: str, instruction: str = GRPO_SYSTEM_PROMPT) -> str:
    """The SFT/GRPO flat-text prompt up to and including the note.

    Uses GRPO_SYSTEM_PROMPT, whose rule 8 places the tag after the XML. Few-shot
    examples are omitted, matching the evaluation-time prompt.
    """
    return format_preference_prompt(note, instruction, "NONE")


def build_verifier_prompt(note: str, candidate_xml: str | None) -> str:
    """Prompt for the verification pass: candidate XML is given, and the model
    is left to write only the verdict after ``<qc_status>``."""
    return f"{extraction_prefix(note)}{canonicalize(candidate_xml)}\n{QC_OPEN}"


def verdict_completion(is_complete: bool) -> str:
    return f"{COMPLETE if is_complete else INCOMPLETE}{QC_CLOSE}"


def build_single_pass_example(note: str, gold_xml: str) -> dict[str, Any]:
    """Prompt/completion pair teaching the tag *after* the model's own XML.

    Needed only for single-pass use: in prefill examples ``\\n<qc_status>`` sits
    in the prompt, so without these the model never learns to emit it after
    ``</phi_entities>``. The gold is complete by definition.
    """
    return {
        "prompt": extraction_prefix(note),
        "completion": f"{canonicalize(gold_xml)}\n{QC_OPEN}{verdict_completion(True)}",
        "is_complete": True,
    }


def label_candidate(candidate_xml: str | None, gold_xml: str) -> bool:
    """True iff the candidate leaves no reference entity uncovered (zero-leak)."""
    return note_is_complete(candidate_xml or "", gold_xml)


def parse_verdict(text: str | None) -> str | None:
    """'complete' / 'incomplete' / None. Accepts a bare completion
    (``incomplete</qc_status>``) or text containing the full tag."""
    if not isinstance(text, str):
        return None
    m = _VERDICT_RE.match(text)
    if m:
        return m.group(1).lower()
    m = re.search(r"<qc_status>\s*(complete|incomplete)\s*</qc_status>", text, re.I)
    return m.group(1).lower() if m else None


# ─── reward (GRPO) ────────────────────────────────────────────────────────────

# Asymmetric calibration: missing a leak is the dangerous error.
#   claimed \ actual   complete   incomplete
#   complete              1.0        0.0     <- overconfident: leak released
#   incomplete            0.5        1.0     <- over-cautious: wasted review
# The 0.5 fixes the implied operating point. If p = P(note is incomplete), the
# expected reward is 1-p for "complete" and 0.5+0.5p for "incomplete", so the
# reward-maximizing policy flags a note once p > 1/3.
REWARD_TABLE: dict[tuple[str, bool], float] = {
    (COMPLETE, True): 1.0,
    (COMPLETE, False): 0.0,
    (INCOMPLETE, False): 1.0,
    (INCOMPLETE, True): 0.5,
}
FORMAT_WEIGHT = 0.2
CALIBRATION_WEIGHT = 0.8


def _completion_text(c: Any) -> str:
    """TRL passes plain strings for text prompts, message lists for chat."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    return str(c)


def verifier_reward_one(completion: str, is_complete: bool) -> float:
    """Well-formed verdict earns the format share; correctness earns the rest.

    Ordering: correct 1.0 > over-cautious 0.6 > overconfident 0.2 > malformed 0.
    A malformed output earns nothing, so the policy is pushed to always commit
    to a verdict (at inference a missing verdict is treated as incomplete).
    """
    m = _VERDICT_RE.match(completion or "")
    if not m:
        return 0.0
    verdict = m.group(1).lower()
    return FORMAT_WEIGHT + CALIBRATION_WEIGHT * REWARD_TABLE[(verdict, bool(is_complete))]


def qc_verifier_reward(prompts=None, completions=None, is_complete=None, **kwargs) -> list[float]:
    """TRL GRPOTrainer reward function. Reads the ``is_complete`` dataset column."""
    if completions is None or is_complete is None:
        raise ValueError("qc_verifier_reward needs `completions` and the `is_complete` column")
    return [verifier_reward_one(_completion_text(c), bool(y))
            for c, y in zip(completions, is_complete)]


# ─── scoring: P(incomplete) without generation ────────────────────────────────

@dataclass
class _VerdictTokens:
    complete: list[int]
    incomplete: list[int]


def verdict_token_ids(tokenizer) -> _VerdictTokens:
    """Token ids of each verdict *in context* (right after ``<qc_status>``).

    Tokenizing in context matters: BPE merges across the boundary would
    otherwise give ids that never occur at scoring time. Raises if the context
    tokenization shifts when a verdict is appended.
    """
    ctx = f"</phi_entities>\n{QC_OPEN}"
    base = tokenizer(ctx, add_special_tokens=False)["input_ids"]
    out = {}
    for v in (COMPLETE, INCOMPLETE):
        ids = tokenizer(ctx + v, add_special_tokens=False)["input_ids"]
        if ids[: len(base)] != base:
            raise ValueError(f"tokenization of '{QC_OPEN}' shifts when followed by '{v}'")
        out[v] = ids[len(base):]
    if out[COMPLETE][0] == out[INCOMPLETE][0]:
        raise ValueError("verdicts share a first token; single-pass scoring invalid")
    return _VerdictTokens(out[COMPLETE], out[INCOMPLETE])


def _forward_tail_logits(model, ids, att, keep):
    """Logits for the last ``keep`` positions only.

    Full-vocabulary logits at every position would be ~14 GB in float32 for a
    batch of 8 x 3k tokens x 151k vocab; only the last few positions are used.
    ``logits_to_keep`` (``num_logits_to_keep`` before transformers 4.50) avoids
    materializing the rest. Falls back to slicing if neither is supported.
    """
    # Left padding: pass explicit positions so real tokens start at 0, exactly
    # as generate() does. (RoPE attention depends only on relative position, so
    # this is also numerically the same as right padding.)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(input_ids=ids, attention_mask=att, position_ids=pos, **{kw: keep}).logits
        except TypeError:
            continue
    return model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -keep:, :]


def _continuation_logprobs(model, tokenizer, prompts, cont, also_first, *, batch_size, progress):
    """One forward pass per prompt over ``prompt + cont``.

    Returns, per prompt and in input order, ``(log P(cont | prompt),
    log P(also_first | prompt))``. ``also_first`` is read from the same logits
    row that predicts ``cont[0]``, so a second verdict's first token is free.

    Batches are length-sorted (less padding) and left-padded, so every row's
    last ``len(cont)+1`` positions line up and only those logits are computed.

    No truncation: cutting the prompt from the left would remove the
    instruction block (the failure TRL's old 512-token default caused). Prompts
    here run ~1-4k tokens against a 32k context.
    """
    import torch

    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    enc_all = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(enc_all[i]))
    keep = len(cont) + 1          # positions predicting cont[0..] plus the final one
    batches = range(0, len(order), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            batches = tqdm(batches, desc="scoring", leave=False)
        except ImportError:
            pass
    res: list[tuple[float, float] | None] = [None] * len(prompts)
    for s in batches:
        idx = order[s:s + batch_size]
        seqs = [enc_all[i] + cont for i in idx]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):                  # left padding
            ids[r, L - len(x):] = torch.tensor(x)
            att[r, L - len(x):] = 1
        with torch.no_grad():
            tail = _forward_tail_logits(model, ids.to(device), att.to(device), keep)
        logp = torch.log_softmax(tail.float(), dim=-1)
        # tail[:, k] predicts cont[k] for k < len(cont); the last row is unused
        for r, i in enumerate(idx):
            lp = sum(logp[r, k, cont[k]].item() for k in range(len(cont)))
            res[i] = (lp, logp[r, 0, also_first].item())
    return res


def score_incomplete(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int = 4,
    progress: bool = True,
) -> list[float]:
    """P(verdict = incomplete) for each verifier prompt, from log-probabilities.

    The verdicts differ at their first token ("in" vs "complete" for the Qwen3
    tokenizer), so a single pass over ``prompt + incomplete-tokens`` yields the
    full log-probability of "incomplete" *and* the first-token log-probability
    of "complete" from the same row. A multi-token "complete" gets one extra
    pass for its tail. The two sequence probabilities are renormalized against
    each other, so the score is a proper two-way probability even when the model
    puts mass elsewhere. This is the quantity to threshold, calibrate, and sweep
    for the review-budget curve.
    """
    vt = verdict_token_ids(tokenizer)
    inc = _continuation_logprobs(model, tokenizer, prompts, vt.incomplete, vt.complete[0],
                                 batch_size=batch_size, progress=progress)
    if len(vt.complete) == 1:
        comp = [first for _, first in inc]
    else:
        comp = [lp for lp, _ in _continuation_logprobs(
            model, tokenizer, prompts, vt.complete, vt.complete[0],
            batch_size=batch_size, progress=progress)]
    out = []
    for (li, _), lc in zip(inc, comp):
        m = max(li, lc)
        out.append(math.exp(li - m) / (math.exp(li - m) + math.exp(lc - m)))
    return out

