"""Decoding-strategy helpers: greedy vs. K-sample union.

Motivation
----------
Recall in this project is measured with *directional containment* (a GT entity
counts as protected only when a prediction covers the whole span — see
``metrics._covers``). More complete spans therefore raise recall. Sampling once
at temperature > 0 does NOT help: it perturbs the output symmetrically, lengthening
some spans while truncating/corrupting others.

A *union over K samples* is the principled way to spend compute on recall: the
union of detected PHI spans can only grow, so recall is monotone non-decreasing
in K. It costs precision (the union also aggregates a few hallucinated spans),
but for de-identification over-redaction is safe — so it trades exactly the way
the safety metric wants. The cost is K× the forward passes, which is the point of
the ablation: does cheap deterministic QC beat expensive union sampling?

``union_predictions`` merges K raw model outputs for ONE note into a single
<phi_entities> XML that round-trips through ``metrics.extract_entities`` (name
``<variations>`` and date ``<original>/<formatted>`` sub-structure preserved).
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from typing import Any

from ..qc.repair import SCHEMA, empty_phi_xml, repair_xml
from .metrics import _normalize, extract_entities

# parent tag -> singular child tag, in schema order (names, ages, dates, ...)
_PARENTS: tuple[tuple[str, str], ...] = SCHEMA


def _entity_signature(parent_tag: str, child: ET.Element) -> str | None:
    """Stable dedup key for one entity element.

    name  -> all <variations>/<value> texts, normalized and sorted
    date  -> <original> text (falls back to <formatted>)
    other -> element text
    Returns None when the element carries no usable text (skip it).
    """
    if parent_tag == "names":
        variations = child.find("variations")
        vals = []
        if variations is not None:
            vals = [v.text.strip() for v in variations.findall("value")
                    if v.text and v.text.strip()]
        if not vals and child.text and child.text.strip():
            vals = [child.text.strip()]
        if not vals:
            return None
        return "name::" + "|".join(sorted(_normalize(v) for v in vals))

    if parent_tag == "dates":
        orig = child.find("original")
        fmt = child.find("formatted")
        text = None
        if orig is not None and orig.text and orig.text.strip():
            text = orig.text.strip()
        elif fmt is not None and fmt.text and fmt.text.strip():
            text = fmt.text.strip()
        return f"date::{_normalize(text)}" if text else None

    if child.text and child.text.strip():
        return f"{child.tag}::{_normalize(child.text)}"
    return None


def union_predictions(preds: list[str]) -> str:
    """Merge K raw model outputs for one note into a single <phi_entities> XML.

    Each prediction is structurally repaired first (so malformed samples still
    contribute), then entity elements are unioned per parent tag with
    normalized-text dedup. The result parses back to the union of every sample's
    detected entities under ``extract_entities``.
    """
    if not preds:
        return empty_phi_xml()

    merged_root = ET.fromstring(empty_phi_xml())
    merged_parents: dict[str, ET.Element] = {
        p: merged_root.find(p) for p, _ in _PARENTS
    }
    seen: set[str] = set()

    for pred in preds:
        rep = repair_xml(pred)
        root = rep.root
        if root is None:
            continue
        for parent_tag, child_tag in _PARENTS:
            parent = root.find(parent_tag)
            if parent is None:
                continue
            for child in parent.findall(child_tag):
                sig = _entity_signature(parent_tag, child)
                if sig is None or sig in seen:
                    continue
                seen.add(sig)
                merged_parents[parent_tag].append(copy.deepcopy(child))

    xml = ET.tostring(merged_root, encoding="unicode")
    import re
    return re.sub(r"<(\w+)></\1>", r"<\1/>", xml)


def union_inproc(
    model: Any,
    notes: list[str],
    *,
    k: int = 5,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """In-process greedy / K-sample-union inference via a QwenDeidBaseline.

    ``temperature == 0`` → one greedy pass per note (``predict_xml_batch``, k
    ignored). ``temperature > 0`` → ``k`` sampled draws per note unioned into one
    <phi_entities> XML. Returns one prediction per note, same order. No vLLM.
    """
    from tqdm import tqdm

    out: list[str] = []
    batches = [notes[i:i + batch_size] for i in range(0, len(notes), batch_size)]
    greedy = temperature == 0
    desc = "greedy" if greedy else f"union(K={k},T={temperature})"
    for batch in tqdm(batches, desc=desc):
        if greedy:
            out.extend(model.predict_xml_batch(batch))
        else:
            for samples in model.sample_xml_batch(batch, k=k, temperature=temperature):
                out.append(union_predictions(samples))
    return out


def make_qc_config(use_tier3: bool = True, v2: bool = False):
    """Build a QCConfig, auto-degrading Tier-3 → Tier-2 when spaCy is unavailable.

    ``v2`` (2026-10-05) turns on the precision fixes: a missing <qc_status> tag no
    longer escalates every note (the model never emits it, so v1 ran Tier-3 on all
    of them), lenient XML recovery, calendar-valid numeric dates, and the Tier-3
    abbreviation / institution / all-caps guards.

    Tier-3 needs spaCy + ``en_core_web_sm``; without them ``apply_qc_batch``
    raises ImportError on the first escalated sample. Pre-flighting the import
    here keeps notebooks runnable top-to-bottom: Tier-3 is only enabled when it
    actually works, otherwise the deterministic Tier-2 recall floor still runs.
    """
    from ..qc import QCConfig

    if use_tier3:
        try:
            import spacy  # noqa: PLC0415
            spacy.load("en_core_web_sm")
        except Exception as e:  # ImportError (no spaCy) or OSError (no model)
            print(f"  [QC] Tier-3 unavailable ({type(e).__name__}); using Tier-2 only. "
                  f"Install: pip install spacy && python -m spacy download en_core_web_sm")
            use_tier3 = False
    if v2:
        return QCConfig(use_tier3=use_tier3, escalate_on_missing_qc=False, lenient_repair=True,
                        strict_dates=True, precision_guards=True)
    return QCConfig(use_tier3=use_tier3)


def batch_inference_union(
    notes: list[str],
    client: Any,
    model_id: str,
    *,
    k: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    base_seed: int = 2025,
    max_workers: int = 32,
    examples: list[dict] | None = None,
) -> list[str]:
    """K-sample union inference: sample each note K times, union the entities.

    Returns one merged <phi_entities> XML per note (same order as ``notes``).
    Uses K distinct seeds so the samples are genuinely different draws. Set
    ``temperature=0`` to collapse to a single greedy pass (K is ignored, since
    every draw would be identical).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    from .inference import run_vllm_model

    if temperature == 0:
        k = 1  # greedy: all draws identical, no point sampling K times

    results = [""] * len(notes)

    def _process(i: int) -> tuple[int, str]:
        samples = [
            run_vllm_model(
                notes[i], client, model_id,
                temperature=temperature, max_tokens=max_tokens,
                seed=base_seed + j, examples=examples,
            )
            for j in range(k)
        ]
        return i, union_predictions(samples)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, i) for i in range(len(notes))]
        for future in tqdm(as_completed(futures), total=len(notes),
                           desc=f"Union(K={k},T={temperature})"):
            i, result = future.result()
            results[i] = result

    return results
