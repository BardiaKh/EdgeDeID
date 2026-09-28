"""QC calibration: measure how well <qc_status> predicts actual completeness.

The safety-critical number is the False Negative (FN) rate of the QC
classifier — i.e., how often the model confidently claims 'complete' while
PHI was actually missed.  We want FN → 0.

Confusion matrix convention (incomplete = positive):
    TP  claimed=incomplete, actually=incomplete   → correctly flagged
    FP  claimed=incomplete, actually=complete     → over-cautious (safe, harmless)
    TN  claimed=complete,   actually=complete     → correct confidence
    FN  claimed=complete,   actually=incomplete   → DANGEROUS leakage hidden

Per-layer delta tracking
------------------------
Call compute_qc_calibration() once per QC stage (raw model, after Tier-2,
after Tier-3) to produce a QCCalibrationResult at each stage.  The notebook
compares the FN rates across stages to show how each tier reduces the
dangerous overconfidence scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..eval.metrics import note_recall
from ..rl.rewards import QC_COMPLETE_THRESHOLD, _extract_qc_status


@dataclass
class QCCalibrationResult:
    # confusion matrix (incomplete = positive class)
    tp: int = 0   # flagged incomplete, truly incomplete
    fp: int = 0   # flagged incomplete, but was complete (over-cautious)
    tn: int = 0   # confident complete, truly complete
    fn: int = 0   # confident complete, but missed PHI  ← DANGEROUS

    n_missing_tag: int = 0   # samples where <qc_status> tag is absent
    n_total: int = 0
    completeness_threshold: float = QC_COMPLETE_THRESHOLD  # recall cutoff

    # per-sample completeness scores (for histogram plotting in notebook)
    scores: list[float] = field(default_factory=list, repr=False)

    # ── derived rates ──────────────────────────────────────────────────────────

    @property
    def fn_rate(self) -> float:
        """P(claimed=complete | actually=incomplete) — the dangerous miss rate.
        Minimize this. Ideal = 0."""
        denom = self.fn + self.tp
        return self.fn / denom if denom else 0.0

    @property
    def fp_rate(self) -> float:
        """P(claimed=incomplete | actually=complete) — over-caution rate.
        Safe to have high; low means model is confident when it should be."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def qc_recall(self) -> float:
        """TP / (TP + FN) — of truly-incomplete notes, fraction correctly flagged."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def qc_precision(self) -> float:
        """TP / (TP + FP) — when flagged, how often is it actually incomplete."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def tag_coverage(self) -> float:
        """Fraction of samples that produced a <qc_status> tag."""
        return (self.n_total - self.n_missing_tag) / self.n_total if self.n_total else 0.0

    @property
    def escalation_rate(self) -> float:
        """Fraction of samples the model flagged as incomplete (TP + FP)."""
        return (self.tp + self.fp) / self.n_total if self.n_total else 0.0

    @property
    def n_actually_incomplete(self) -> int:
        return self.fn + self.tp

    @property
    def n_actually_complete(self) -> int:
        return self.tn + self.fp


def compute_qc_calibration(
    predictions: list[str],
    ground_truths: list[str],
    completeness_threshold: float = QC_COMPLETE_THRESHOLD,
) -> QCCalibrationResult:
    """Measure calibration of <qc_status> against actual per-sample recall.

    'Actually complete' is defined as per-note recall >= completeness_threshold,
    where recall is ``note_recall`` — the same-category, full-containment rule
    behind the reported zero-leak metric. With the default threshold of 1.0,
    "actually complete" is exactly "zero-leak note", so a calibration table can
    be read directly against the zero-leak numbers. (This previously used the
    bidirectional ``entity_recall``, under which a partial span such as "Amit"
    for "Amit Kumar" counted as caught even though zero-leak counts it a leak.)

    Missing <qc_status> tags are treated as claimed=incomplete (conservative),
    but counted separately so you can see tag coverage.
    """
    result = QCCalibrationResult(completeness_threshold=completeness_threshold)
    result.n_total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        status = _extract_qc_status(pred)
        score = note_recall(pred, gt)
        result.scores.append(score)

        actually_complete = score >= completeness_threshold

        if status is None:
            result.n_missing_tag += 1
            claimed_incomplete = True  # conservative: no tag → treat as uncertain
        else:
            claimed_incomplete = status == "incomplete"

        if claimed_incomplete and not actually_complete:
            result.tp += 1
        elif claimed_incomplete and actually_complete:
            result.fp += 1
        elif not claimed_incomplete and actually_complete:
            result.tn += 1
        else:
            result.fn += 1  # claimed complete but actually incomplete — DANGEROUS

    return result


def print_calibration_report(r: QCCalibrationResult, title: str = "QC CALIBRATION") -> None:
    w = 58
    print("=" * w)
    print(f" {title}")
    print("=" * w)
    print(f"  Completeness threshold : recall >= {r.completeness_threshold}")
    print(f"  Total samples          : {r.n_total}")
    print(f"  Tag coverage           : {r.tag_coverage:.4f}  ({r.n_total - r.n_missing_tag}/{r.n_total} have <qc_status>)")
    print(f"  Escalation rate        : {r.escalation_rate:.4f}  (model flagged incomplete)")
    print()
    print("  Confusion Matrix  (incomplete = positive):")
    print(f"    TP  flagged, truly incomplete   : {r.tp:5d}")
    print(f"    FP  flagged, but was complete   : {r.fp:5d}  ← over-cautious (safe)")
    print(f"    TN  confident, truly complete   : {r.tn:5d}")
    print(f"    FN  confident, but missed PHI   : {r.fn:5d}  ← DANGEROUS")
    print()
    print(f"  QC Recall   TP/(TP+FN)            : {r.qc_recall:.4f}  (catch rate for real misses)")
    print(f"  QC Precision TP/(TP+FP)           : {r.qc_precision:.4f}")
    print(f"  FN rate     (hidden leakage)      : {r.fn_rate:.4f}  ← minimize this")
    print(f"  FP rate     (over-caution)        : {r.fp_rate:.4f}")
    print("=" * w)


def calibration_to_dict(r: QCCalibrationResult, label: str = "") -> dict:
    """Compact dict for DataFrame rows / comparison tables."""
    return {
        "label": label,
        "n_total": r.n_total,
        "tag_coverage": round(r.tag_coverage, 4),
        "escalation_rate": round(r.escalation_rate, 4),
        "qc_recall": round(r.qc_recall, 4),
        "qc_precision": round(r.qc_precision, 4),
        "fn_rate": round(r.fn_rate, 4),
        "fp_rate": round(r.fp_rate, 4),
        "tp": r.tp, "fp": r.fp, "tn": r.tn, "fn": r.fn,
        "n_missing_tag": r.n_missing_tag,
    }
