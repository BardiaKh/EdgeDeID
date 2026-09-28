"""Metrics for the QC verifier.

Convention: positive class = INCOMPLETE (the note leaks and needs review);
``p`` = P(incomplete). The operational question for a review filter is:

  if every note with p >= t goes to a human, what share of notes is reviewed
  (flag rate), what share of leaky notes is caught (catch rate), and what share
  of the notes released WITHOUT review still leak (residual leak rate)?

``budget_curve`` answers it across all thresholds — the headline figure.
Tag *coverage* is not a metric here: always answering "complete" has 100%
coverage and catches nothing.
"""

from __future__ import annotations

import math
from typing import Sequence

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auroc(y: Sequence[int], s: Sequence[float]) -> float | None:
    """ROC AUC (positive = incomplete); None if only one class is present."""
    return float(roc_auc_score(y, s)) if 0 < sum(y) < len(y) else None


def average_precision(y: Sequence[int], s: Sequence[float]) -> float | None:
    """Area under the precision-recall curve (positive = incomplete)."""
    return float(average_precision_score(y, s)) if sum(y) else None


def brier(y: Sequence[int], p: Sequence[float]) -> float:
    return float(brier_score_loss(y, p))


def ece(y: Sequence[int], p: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error: bin-weighted |observed rate - mean prediction|."""
    n = len(y)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, pi in enumerate(p) if (lo <= pi < hi) or (b == bins - 1 and pi == 1.0)]
        if idx:
            conf = sum(p[i] for i in idx) / len(idx)
            acc = sum(y[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def operating_point(y: Sequence[int], p: Sequence[float], t: float) -> dict[str, float]:
    """Review-filter behaviour when every note with p >= t is sent to a human."""
    n = len(y)
    flag = [pi >= t for pi in p]
    n_flag = sum(flag)
    n_pos = sum(y)
    tp = sum(f and yi for f, yi in zip(flag, y))
    released = n - n_flag
    leak_released = sum((not f) and yi for f, yi in zip(flag, y))
    return {
        "threshold": t,
        "flag_rate": n_flag / n,                                  # review budget
        "catch_rate": tp / n_pos if n_pos else math.nan,          # leaky notes caught
        "flag_precision": tp / n_flag if n_flag else math.nan,    # reviews that were needed
        "residual_leak_rate": leak_released / released if released else 0.0,
        "base_leak_rate": n_pos / n,                              # no filter at all
    }


def budget_curve(y: Sequence[int], p: Sequence[float], n_points: int = 101) -> list[dict]:
    """Operating points over thresholds 0..1 (plus every distinct score)."""
    ts = sorted(set([i / (n_points - 1) for i in range(n_points)] + list(p)))
    return [operating_point(y, p, t) for t in ts]


def threshold_for_catch_rate(y: Sequence[int], p: Sequence[float], target: float) -> float:
    """Highest threshold whose catch rate reaches ``target`` (e.g. 0.95).

    Pick this on the synthetic *validation* split and apply it unchanged to the
    test sets; choosing it on test data would overstate the filter.
    """
    best = 0.0
    for t in sorted(set(p), reverse=True):
        if operating_point(y, p, t)["catch_rate"] >= target:
            best = t
            break
    return best


def summarize(y: Sequence[int], p: Sequence[float], thresholds: dict[str, float]) -> dict:
    out = {
        "n": len(y),
        "base_leak_rate": sum(y) / len(y) if y else math.nan,
        "auroc": auroc(y, p),
        "auprc": average_precision(y, p),
        "ece": ece(y, p),
        "brier": brier(y, p),
    }
    for name, t in thresholds.items():
        for k, v in operating_point(y, p, t).items():
            if k != "base_leak_rate":
                out[f"{k}@{name}"] = v
    return out
