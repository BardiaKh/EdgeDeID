"""Evaluation metrics for PHI de-identification.

Matches the entity-matching logic used in the eval notebooks (bidirectional
substring matching) so that numbers here are comparable to reported results.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_PHI_PARENT_TAGS = [
    "names", "ages", "dates", "places", "addresses",
    "phones", "emails", "ids", "urls", "devices",
]
_PHI_ENTITY_TAGS = [t.rstrip("s") if t != "addresses" else "address" for t in _PHI_PARENT_TAGS]
# manual override for irregular plurals
_TAG_SINGULAR = {
    "names": "name", "ages": "age", "dates": "date", "places": "place",
    "addresses": "address", "phones": "phone", "emails": "email",
    "ids": "id", "urls": "url", "devices": "device",
}


# ─── XML parsing ──────────────────────────────────────────────────────────────

def _extract_phi_block(text: str) -> str | None:
    m = re.search(r"<phi_entities\b[^>]*>.*?</phi_entities>", text, re.DOTALL | re.IGNORECASE)
    return m.group() if m else None


def extract_entities(xml_str: str) -> dict[str, list[str]]:
    """Parse a <phi_entities> XML string into {entity_type: [text, …]}.

    Replicates the extraction logic from the eval notebooks, including:
    - name variations (all <value> children)
    - formatted date normalisation (YYYY-MM-DD only; falls back to original)
    """
    entities: dict[str, list[str]] = {}

    if not isinstance(xml_str, str):
        return entities

    # allow passing the full completion text
    block = _extract_phi_block(xml_str)
    if block is not None:
        xml_str = block

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return entities

    for parent_tag in _PHI_PARENT_TAGS:
        singular = _TAG_SINGULAR[parent_tag]
        parent_elem = root.find(parent_tag)
        if parent_elem is None:
            continue
        for child in parent_elem.findall(singular):
            if singular == "name":
                variations = child.find("variations")
                if variations is not None:
                    for val in variations.findall("value"):
                        if val.text and val.text.strip():
                            entities.setdefault(singular, []).append(val.text.strip())
            elif singular == "date":
                # Match dates on the raw source span (`original`), not the
                # dataset's ISO `formatted` value. NER baselines emit the
                # detected span verbatim ("12-25-2028"), so comparing against
                # the normalized ISO form ("2028-12-25") spuriously failed even
                # when the date was correctly detected. Detection is what the
                # PHI-leak metric should measure; `formatted` is a fallback for
                # entries that lack an `original`.
                orig = child.find("original")
                fmt = child.find("formatted")
                text = None
                if orig is not None and orig.text and orig.text.strip():
                    text = orig.text.strip()
                elif fmt is not None and fmt.text and fmt.text.strip():
                    text = fmt.text.strip()
                if text:
                    entities.setdefault(singular, []).append(text)
            else:
                if child.text and child.text.strip():
                    entities.setdefault(singular, []).append(child.text.strip())

    return entities


def _soft_match(a: str, b: str) -> bool:
    """Bidirectional substring match (case-insensitive)."""
    a, b = a.lower(), b.lower()
    return a in b or b in a


# ─── per-sample comparison ────────────────────────────────────────────────────

def compare_sample(
    gt_xml: str,
    pred_xml: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Compare one prediction against one ground truth.

    Returns (gt_stats, pred_stats) where each maps tag → {correct, total}.
    gt_stats   drives recall  (how many GT entities were found)
    pred_stats drives precision (how many predicted entities are correct)
    """
    gt_entities = extract_entities(gt_xml)
    pred_entities = extract_entities(pred_xml)

    gt_stats: dict[str, dict] = {}
    for tag, gt_vals in gt_entities.items():
        pred_vals = pred_entities.get(tag, [])
        correct = sum(
            1 for v in gt_vals if any(_soft_match(v, p) for p in pred_vals)
        )
        gt_stats[tag] = {"correct": correct, "total": len(gt_vals)}

    pred_stats: dict[str, dict] = {}
    for tag, pred_vals in pred_entities.items():
        gt_vals = gt_entities.get(tag, [])
        correct = sum(
            1 for p in pred_vals if any(_soft_match(p, g) for g in gt_vals)
        )
        pred_stats[tag] = {"correct": correct, "total": len(pred_vals)}

    return gt_stats, pred_stats


# ─── aggregate metrics ────────────────────────────────────────────────────────

@dataclass
class TagMetrics:
    precision: float
    recall: float
    f1: float
    gt_count: int
    pred_count: int
    tp: int


@dataclass
class EvalResult:
    """Full evaluation result for a set of predictions, including overall and per-entity metrics."""

    # overall
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float          # mean of per-class F1 (unweighted)
    f_beta: float            # F-β (β=2): recall-weighted F1 for safety-critical evaluation

    # reliability & safety
    xml_parse_rate: float    # fraction with valid <phi_entities> XML
    hallucination_rate: float  # 1 − micro_precision (FP pressure on safety)
    token_reduction_rate: float  # fraction of source tokens that are PHI
    zero_leak_rate: float    # fraction of notes where every GT entity was found (recall == 1.0)

    # per-entity breakdown
    per_entity: dict[str, TagMetrics] = field(default_factory=dict)

    # raw counts
    total_tp: int = 0
    total_gt: int = 0
    total_pred: int = 0
    n_samples: int = 0


def _prf(tp: int, total_pred: int, total_gt: int) -> tuple[float, float, float]:
    p = tp / total_pred if total_pred else 0.0
    r = tp / total_gt if total_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def compute_metrics(
    predictions: list[str],
    ground_truths: list[str],
    source_notes: list[str] | None = None,
    supported_entities: set[str] | None = None,
) -> EvalResult:
    """Compute all metrics over a list of predictions.

    Parameters
    ----------
    predictions  : raw model output strings (may or may not contain valid XML)
    ground_truths: corresponding ground-truth XML strings
    source_notes : optional raw notes (needed for token_reduction_rate)
    supported_entities : if given, the schema keys this model is *capable* of
        emitting. Only these entities are included in the per-entity breakdown
        and the macro-F1 average — entities outside this set show as N/A rather
        than a misleading 0.0. Micro-F1, recall, and zero-leak are deliberately
        left full-coverage: PHI the model structurally cannot detect is still a
        real leak, so it must count against the safety metrics.
    """
    assert len(predictions) == len(ground_truths)

    # xml_parse_rate
    parsed = [_extract_phi_block(p) is not None for p in predictions]
    xml_parse_rate = sum(parsed) / len(predictions) if predictions else 0.0

    # accumulate per-tag stats + zero-leak tracking
    accum_gt: dict[str, dict] = {}
    accum_pred: dict[str, dict] = {}
    n_zero_leak = 0

    for pred, gt in zip(predictions, ground_truths):
        gt_s, pred_s = compare_sample(gt, pred)
        for tag, stats in gt_s.items():
            accum_gt.setdefault(tag, {"correct": 0, "total": 0})
            accum_gt[tag]["correct"] += stats["correct"]
            accum_gt[tag]["total"] += stats["total"]
        for tag, stats in pred_s.items():
            accum_pred.setdefault(tag, {"correct": 0, "total": 0})
            accum_pred[tag]["correct"] += stats["correct"]
            accum_pred[tag]["total"] += stats["total"]
        # zero-leak: note has no missed PHI if every GT entity was matched
        sample_gt  = sum(v["total"]   for v in gt_s.values())
        sample_tp  = sum(v["correct"] for v in gt_s.values())
        if sample_gt == 0 or sample_tp == sample_gt:
            n_zero_leak += 1

    zero_leak_rate = n_zero_leak / len(predictions) if predictions else 0.0

    # per-entity TagMetrics — restricted to supported labels (macro fairness);
    # micro/recall/zero-leak below stay full-coverage for safety honesty.
    all_tags = set(accum_gt) | set(accum_pred)
    report_tags = all_tags & supported_entities if supported_entities is not None else all_tags
    per_entity: dict[str, TagMetrics] = {}
    macro_f1_sum = 0.0

    for tag in report_tags:
        tp = accum_gt.get(tag, {}).get("correct", 0)
        total_gt = accum_gt.get(tag, {}).get("total", 0)
        total_pred = accum_pred.get(tag, {}).get("total", 0)
        p, r, f = _prf(tp, total_pred, total_gt)
        per_entity[tag] = TagMetrics(
            precision=p, recall=r, f1=f,
            gt_count=total_gt, pred_count=total_pred, tp=tp,
        )
        macro_f1_sum += f

    macro_f1 = macro_f1_sum / len(report_tags) if report_tags else 0.0

    # micro overall — full coverage across every GT/pred tag, regardless of
    # supported_entities, so undetectable PHI still counts against the model.
    total_tp = sum(v.get("correct", 0) for v in accum_gt.values())
    total_gt_all = sum(v.get("total", 0) for v in accum_gt.values())
    total_pred_all = sum(v.get("total", 0) for v in accum_pred.values())
    micro_p, micro_r, micro_f = _prf(total_tp, total_pred_all, total_gt_all)

    hallucination_rate = 1.0 - micro_p

    # F-β with β=2: weights recall twice as much as precision
    _b2 = 4  # β² = 2² = 4
    f_beta = (1 + _b2) * micro_p * micro_r / (_b2 * micro_p + micro_r) if (_b2 * micro_p + micro_r) else 0.0

    # token_reduction_rate
    if source_notes is not None:
        rates = []
        for note, pred in zip(source_notes, predictions):
            note_tokens = note.split()
            ents = extract_entities(pred)
            phi_tokens = sum(len(v.split()) for vals in ents.values() for v in vals)
            rate = phi_tokens / len(note_tokens) if note_tokens else 0.0
            rates.append(min(rate, 1.0))
        token_reduction_rate = sum(rates) / len(rates)
    else:
        token_reduction_rate = float("nan")

    return EvalResult(
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f,
        macro_f1=macro_f1,
        f_beta=f_beta,
        xml_parse_rate=xml_parse_rate,
        hallucination_rate=hallucination_rate,
        token_reduction_rate=token_reduction_rate,
        zero_leak_rate=zero_leak_rate,
        per_entity=per_entity,
        total_tp=total_tp,
        total_gt=total_gt_all,
        total_pred=total_pred_all,
        n_samples=len(predictions),
    )


# ─── reporting helpers ────────────────────────────────────────────────────────

def print_report(result: EvalResult, title: str = "EVALUATION REPORT") -> None:
    """Print a formatted report suitable for logging verification."""
    w = 60
    print("=" * w)
    print(f" {title}")
    print("=" * w)
    print(f"  Samples evaluated : {result.n_samples}")
    print(f"  XML parse rate    : {result.xml_parse_rate:.4f}")
    print()
    print(f"  Micro Precision   : {result.micro_precision:.4f}")
    print(f"  Micro Recall      : {result.micro_recall:.4f}")
    print(f"  Micro F1          : {result.micro_f1:.4f}")
    print(f"  F-β (β=2)         : {result.f_beta:.4f}  (recall-weighted)")
    print(f"  Macro F1          : {result.macro_f1:.4f}")
    print()
    print(f"  Zero-leak rate    : {result.zero_leak_rate:.4f}  (notes with no missed PHI)")
    print(f"  Hallucination rate: {result.hallucination_rate:.4f}  (1 − Precision)")
    if not (result.token_reduction_rate != result.token_reduction_rate):  # nan check
        print(f"  Token reduction   : {result.token_reduction_rate:.4f}")
    print()
    print(f"{'Tag':<12} {'P':>8} {'R':>8} {'F1':>8} {'GT':>8} {'Pred':>8}")
    print("-" * w)
    for tag, m in sorted(result.per_entity.items()):
        print(
            f"{tag:<12} {m.precision:>8.4f} {m.recall:>8.4f} {m.f1:>8.4f}"
            f" {m.gt_count:>8d} {m.pred_count:>8d}"
        )
    print("=" * w)


def to_dataframe(result: EvalResult):
    """Return a pandas DataFrame of per-entity metrics."""
    import pandas as pd

    rows = []
    for tag, m in sorted(result.per_entity.items()):
        rows.append({
            "entity": tag,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "gt_count": m.gt_count,
            "pred_count": m.pred_count,
        })
    # append aggregate row
    rows.append({
        "entity": "micro",
        "precision": round(result.micro_precision, 4),
        "recall": round(result.micro_recall, 4),
        "f1": round(result.micro_f1, 4),
        "gt_count": result.total_gt,
        "pred_count": result.total_pred,
    })
    rows.append({
        "entity": "macro",
        "precision": None,
        "recall": None,
        "f1": round(result.macro_f1, 4),
        "gt_count": None,
        "pred_count": None,
    })
    return pd.DataFrame(rows)