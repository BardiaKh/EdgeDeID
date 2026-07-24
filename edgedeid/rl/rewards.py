"""Reward functions for PHI de-identification GRPO training."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_PHI_TAGS = ["name", "age", "date", "place", "address", "phone", "email", "id", "url", "device"]


# ─── XML helpers ──────────────────────────────────────────────────────────────

def _extract_phi_xml(text: str) -> str | None:
    """Pull <phi_entities>…</phi_entities> block out of a completion."""
    m = re.search(r"<phi_entities\b[^>]*>.*?</phi_entities>", text, re.DOTALL | re.IGNORECASE)
    return m.group() if m else None


def _parse_phi_xml(xml_str: str) -> ET.Element | None:
    try:
        return ET.fromstring(xml_str)
    except ET.ParseError:
        return None


def _extract_entity_texts(root: ET.Element) -> list[str]:
    """Collect all leaf text values from a parsed phi_entities element."""
    texts: list[str] = []
    for tag in _PHI_TAGS:
        for elem in root.iter(tag):
            if tag == "name":
                for val in elem.iter("value"):
                    if val.text and val.text.strip():
                        texts.append(val.text.strip())
            elif tag == "date":
                orig = elem.find("original")
                if orig is not None and orig.text and orig.text.strip():
                    texts.append(orig.text.strip())
            else:
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
    return texts


def _extract_qc_status(text: str) -> str | None:
    """Extract the <qc_status> tag content (complete / incomplete) from a completion."""
    m = re.search(r"<qc_status\s*>(complete|incomplete)</qc_status>", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


# ─── individual reward functions ──────────────────────────────────────────────

def reward_xml_structure(completion: str) -> float:
    """1.0 if the completion contains valid, parseable <phi_entities> XML; else 0.0.

    This is the hard gate: if 0.0, no subsequent reward is computed.
    """
    xml_str = _extract_phi_xml(completion)
    if xml_str is None:
        return 0.0
    return 1.0 if _parse_phi_xml(xml_str) is not None else 0.0


def reward_phi_text_presence(completion: str, source_note: str) -> float:
    """Fraction of tagged entities that actually appear verbatim in the source note.

    Penalises hallucinated PHI that was never in the original text.
    """
    xml_str = _extract_phi_xml(completion)
    if xml_str is None:
        return 0.0
    root = _parse_phi_xml(xml_str)
    if root is None:
        return 0.0

    entities = _extract_entity_texts(root)
    if not entities:
        return 1.0  # empty tags with no hallucinations is correct

    note_lower = source_note.lower()
    # entity must appear as a substring of the note (not the other way around —
    # we want the tagged span to physically exist in the text)
    found = sum(1 for e in entities if e.lower() in note_lower)
    return found / len(entities)  # fraction; consistent with recall-side eval logic


_MIN_MATCH_LEN = 3  # guard against 1–2 char values substring-matching everything


def _soft_match(a: str, b: str) -> bool:
    """Bidirectional substring match (case-insensitive) with a length guard.

    Consistent with the eval notebook's compare_entities logic
    ("a in b or b in a" — catches partial names, date format variants), but the
    shorter operand must be >= _MIN_MATCH_LEN characters unless the two are
    exactly equal. This closes the reward-hacking channel where emitting a
    single-character entity value soft-matches arbitrary ground-truth spans.
    """
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < _MIN_MATCH_LEN:
        return False
    return a in b or b in a


def _entity_list(text: str) -> list[str]:
    """All leaf entity values in the <phi_entities> block of ``text``."""
    xml_str = _extract_phi_xml(text)
    if xml_str is None:
        return []
    root = _parse_phi_xml(xml_str)
    if root is None:
        return []
    return _extract_entity_texts(root)


def reward_completeness(completion: str, ground_truth: str, beta: float = 2.0) -> float:
    """Entity-level F-beta using guarded bidirectional substring matching.

    Uses the same matching logic as the evaluation notebooks so the training
    signal stays consistent with reported metrics, but defaults to F-beta with
    beta=2 (recall-weighted): a missed PHI entity (leakage) is clinically far
    worse than an over-tag, so recall is weighted above precision.
    """
    pred_list = _entity_list(completion)
    true_list = _entity_list(ground_truth)

    if not true_list and not pred_list:
        return 1.0
    if not true_list or not pred_list:
        return 0.0

    # Recall: fraction of GT entities matched by at least one prediction
    recall_hits = sum(
        1 for gt in true_list if any(_soft_match(gt, p) for p in pred_list)
    )
    recall = recall_hits / len(true_list)

    # Precision: fraction of predicted entities matched by at least one GT
    precision_hits = sum(
        1 for p in pred_list if any(_soft_match(p, gt) for gt in true_list)
    )
    precision = precision_hits / len(pred_list)

    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_recall(completion: str, ground_truth: str) -> float:
    """Fraction of ground-truth entities matched by at least one prediction.

    This is the ground truth for the <qc_status> tag: 'complete' means "I did
    not miss any PHI", which is a pure recall question. reward_completeness is
    F-beta and therefore precision-sensitive — an output that catches every
    entity but over-tags scores well below 1.0, so it must NOT be used to
    decide whether the model was actually complete. Over-tagging is the safe
    direction and is penalised by the completeness/presence terms instead.
    """
    pred_list = _entity_list(completion)
    true_list = _entity_list(ground_truth)

    if not true_list:
        return 1.0  # nothing to miss
    if not pred_list:
        return 0.0
    hits = sum(1 for gt in true_list if any(_soft_match(gt, p) for p in pred_list))
    return hits / len(true_list)


# Single source of truth for "the model was actually complete", shared by the
# GRPO reward and the calibration eval. Strict by design: one missed entity is
# one leaked identifier, so anything below full recall counts as incomplete.
QC_COMPLETE_THRESHOLD: float = 1.0


def reward_qc_calibration(
    completion: str,
    recall_score: float,
    threshold: float = QC_COMPLETE_THRESHOLD,
) -> float:
    """Asymmetric metacognitive-calibration reward for the <qc_status> tag.

    ``recall_score`` must come from entity_recall(), not reward_completeness().

    Rewards the model for knowing what it doesn't know, but the four cases are
    NOT symmetric — for a PHI de-identification safety task, confidently
    claiming "complete" while entities were actually missed is the dangerous
    error and is punished hardest:

      claimed   | actual      | reward | rationale
      ----------|-------------|--------|-------------------------------------
      complete  | complete    |  1.0   | correct & confident
      incomplete| incomplete  |  1.0   | correctly flags its own miss
      incomplete| complete    |  0.5   | over-cautious — safe but imprecise
      complete  | incomplete  |  0.0   | overconfident leakage — worst case

    Returns 0.0 if no <qc_status> tag is present (the tag is mandatory).
    """
    status = _extract_qc_status(completion)
    if status is None:
        return 0.0
    actually_complete = recall_score >= threshold
    claimed_complete = status == "complete"
    if claimed_complete == actually_complete:
        return 1.0
    if not claimed_complete and actually_complete:
        return 0.5  # over-cautious: safe but loses precision
    return 0.0  # claimed complete but was not: dangerous overconfidence


def reward_length_penalty(completion: str, soft_max_tokens: int = 1024) -> float:
    """1.0 for outputs within the token budget; decays linearly beyond it.

    Prevents the model from generating redundant or padded content. The exact
    tokenizer is unavailable inside the reward fn, so token count is estimated
    as ~len(text)/4 (a stable English+XML proxy) rather than whitespace splits,
    which badly under-count the whitespace-sparse <phi_entities> XML.
    """
    n_tokens = max(len(completion.split()), len(completion) // 4)
    if n_tokens <= soft_max_tokens:
        return 1.0
    excess = n_tokens - soft_max_tokens
    return max(0.0, 1.0 - excess / soft_max_tokens)


# ─── combined reward ──────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: dict[str, float] = {
    "presence": 0.35,
    "completeness": 0.35,
    "qc": 0.15,
    "length": 0.15,
}


def compute_grpo_reward(
    completion: str,
    source_note: str,
    ground_truth: str,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Sequential composite reward for GRPO PHI de-identification.

    Structure validity is a hard gate: reward is 0 if XML cannot be parsed.
    All other sub-rewards are weighted and summed.
    """
    w = weights or _DEFAULT_WEIGHTS

    if reward_xml_structure(completion) == 0.0:
        return 0.0

    r_presence = reward_phi_text_presence(completion, source_note)
    r_complete = reward_completeness(completion, ground_truth)
    # qc is graded against pure recall, not the F-beta completeness score:
    # "complete" is a claim about missed entities only (see entity_recall).
    r_qc = reward_qc_calibration(completion, entity_recall(completion, ground_truth))
    r_length = reward_length_penalty(completion)

    return (
        w.get("presence", 0.35) * r_presence
        + w.get("completeness", 0.35) * r_complete
        + w.get("qc", 0.15) * r_qc
        + w.get("length", 0.15) * r_length
    )


def make_reward_fn(
    source_notes: list[str] | None = None,
    ground_truths: list[str] | None = None,
    weights: dict[str, float] | None = None,
):
    """Return a TRL-compatible reward function closure.

    The returned function matches the GRPOTrainer signature:
        fn(completions: list[str], **kwargs) -> list[float]

    Dataset columns 'source_note' and 'ground_truth' are read from kwargs when
    source_notes / ground_truths are not provided at construction time.
    """

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        notes = source_notes if source_notes is not None else kwargs.get("source_note", [])
        truths = ground_truths if ground_truths is not None else kwargs.get("ground_truth", [])
        return [
            compute_grpo_reward(c, n, g, weights=weights)
            for c, n, g in zip(completions, notes, truths)
        ]

    return reward_fn


# ─── legacy stubs kept for import compatibility ───────────────────────────────

def reward_exact_xml_match(predicted: str, target: str) -> float:
    """Exact XML-output match. Delegates to reward_completeness (F1 = 1 iff exact)."""
    return reward_completeness(predicted, target)


def reward_entity_f1(predicted: str, target: str) -> float:
    """Entity-level F1. Alias for reward_completeness."""
    return reward_completeness(predicted, target)


def reward_boundary_overlap(predicted: str, target: str) -> float:
    """Token-boundary overlap on PHI spans. Alias for reward_completeness."""
    return reward_completeness(predicted, target)


def reward_phi_schema_compliance(output_text: str) -> float:
    """XML schema compliance check. Alias for reward_xml_structure."""
    return reward_xml_structure(output_text)
