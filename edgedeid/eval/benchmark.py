"""Load the 100-note in-domain benchmark as (notes, ground-truth XML).

The benchmark is ``evals/ext_eval/LLM_Eval.xlsx`` (committed 2026-05-03 as
``evals/synth_eval/LLM Eval.xlsx``, removed in 0f8f188, restored here):
``default_response_0`` is the raw note fed to the model, ``Revision`` is the
same note with inline PHI tags. It replaces ``revizied_evals.xlsx``, a draft
whose model-input column carries annotation tags (15 notes reach the model
with ``<date>`` in them) and whose gold nests tags (``<date><date>``) and puts
possessives inside names ("Hassan's"); 32 of the 34 misses it reported for
EdgeDeID-SFT were those artefacts.

Two repairs are applied to the gold, both counted in the returned stats:

* Misspelled tags (3 in the file): ``<divice>`` -> ``<device>``,
  ``</datede>`` / ``</dated>`` -> ``</date>``, and a doubled ``<date><date>``
  collapsed. Unrepaired, each silently drops a gold entity.
* Gold dates are re-anchored to the note's own wording. The annotator wrote
  dates in ISO form (``On <date>2022-01-15</date>``) where the note says
  "January 15, 2022". Since 8b2876e the metric matches dates on ``<original>``
  -- the raw span, which is what NER baselines emit and what our model is
  trained to copy -- so an ISO ``original`` makes every correct date a miss
  and a false positive. Each gold date whose ``original`` is not verbatim in
  the note is matched to a date expression in the note with the same value;
  that span becomes ``original`` and the ISO value stays in ``formatted``.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

DEFAULT_PATH = "evals/ext_eval/LLM_Eval.xlsx"

_TAG_REPAIRS = [
    (re.compile(r"<date>\s*<date>"), "<date>"),          # doubled opening (row 79)
    (re.compile(r"</date>\s*</dated>"), "</date>"),      # its doubled, misspelled close
    (re.compile(r"</date[a-z]+\s*>"), "</date>"),        # </datede>, </dated>
    (re.compile(r"<divice>"), "<device>"),
]
_WRAPPERS = ["<anonymized_report>", "</anonymized_report>", "<anonymized_report/>",
             "<raw_report>", "</raw_report>", "<raw_report/>"]

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
           "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_DATE_SPAN = re.compile(
    rf"\b(?:{_MONTHS})\.?\s*\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*\d{{4}}\b"   # January 15, 2022 / "March15 ,2023"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\.?,?\s+\d{{4}}\b"  # 15 January 2022
    rf"|\b(?:{_MONTHS})\.?\s+\d{{4}}\b"                                    # January 2022
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"                  # January 15
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b"                                          # 2022-01-15
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"                                 # 01/15/2022
    rf"|\b(?:{_MONTHS})\b",                                               # bare "March"
    re.IGNORECASE)


def _iso(span: str) -> str | None:
    """'YYYY-MM-DD' for a date expression, with '??' / '????' for absent parts."""
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", span.strip(), flags=re.I).replace(",", " ")
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bof\s+", "", s, flags=re.I).replace(".", "")
    s = re.sub(r"\bSept\b", "Sep", s, flags=re.I)
    for fmt, has in [("%B %d %Y", "ymd"), ("%b %d %Y", "ymd"), ("%d %B %Y", "ymd"), ("%d %b %Y", "ymd"),
                     ("%B %Y", "ym"), ("%b %Y", "ym"), ("%B %d", "md"), ("%b %d", "md"),
                     ("%Y-%m-%d", "ymd"), ("%m/%d/%Y", "ymd"), ("%m/%d/%y", "ymd"),
                     ("%m-%d-%Y", "ymd"), ("%m-%d-%y", "ymd"), ("%B", "m"), ("%b", "m")]:
        try:
            d = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return (f"{d.year:04d}" if "y" in has else "????") + "-" + f"{d.month:02d}" + "-" + \
               (f"{d.day:02d}" if "d" in has else "??")
    return None


def _normalise_gold(v: str) -> str:
    """Gold ``formatted`` comes in several spellings: 2023/03/15, 04/10, 02-10."""
    v = v.strip()
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", v):
        return v.replace("/", "-")
    if re.fullmatch(r"\d{2}[/-]\d{2}", v):
        return "????-" + v.replace("/", "-")
    return v


def _same_date(a: str, b: str) -> bool:
    """Equal ignoring positions either side marks unknown with '?'."""
    return len(a) == len(b) and all(x == y or "?" in (x, y) for x, y in zip(a, b))


@dataclass
class BenchmarkStats:
    n_notes: int = 0
    tag_repairs: int = 0
    gold_dates: int = 0
    dates_already_verbatim: int = 0
    dates_reanchored: int = 0
    dates_unanchored: list[tuple[int, str]] = field(default_factory=list)


def _reanchor_dates(gt_xml: str, note: str, row: int, st: BenchmarkStats) -> str:
    root = ET.fromstring(gt_xml)
    spans = [(m.group(0), _iso(m.group(0))) for m in _DATE_SPAN.finditer(note)]
    used: set[str] = set()
    for date in root.iter("date"):
        orig, fmt = date.find("original"), date.find("formatted")
        if orig is None or not (orig.text or "").strip():
            continue
        st.gold_dates += 1
        text = orig.text.strip()
        if text in note:
            st.dates_already_verbatim += 1
            continue
        target = _normalise_gold((fmt.text or "").strip() if fmt is not None else text)
        hits = [s for s, v in spans if v and _same_date(v, target)]
        hit = next((s for s in hits if s not in used), hits[0] if hits else None)
        if hit is None:
            st.dates_unanchored.append((row, text))
            continue
        used.add(hit)
        orig.text = hit
        if fmt is None:
            ET.SubElement(date, "formatted").text = target
        st.dates_reanchored += 1
    return ET.tostring(root, encoding="unicode")


def load_radiology_benchmark(path: str = DEFAULT_PATH, gold_col: str = "Revision",
                             verbose: bool = True) -> tuple[list[str], list[str], BenchmarkStats]:
    from edgedeid.data.preprocessing import preprocess_text
    from edgedeid.data.xml_utils import convert_to_xml

    df = pd.read_excel(path)
    if gold_col != "default_response_1":
        df = df.rename(columns={gold_col: "default_response_1"})
    st = BenchmarkStats(n_notes=len(df))
    gold = df["default_response_1"].astype(str)
    for pat, rep in _TAG_REPAIRS:
        st.tag_repairs += int(gold.str.count(pat.pattern).sum())
        gold = gold.str.replace(pat, rep, regex=True)
    df["default_response_1"] = gold
    with redirect_stdout(io.StringIO()):
        df = preprocess_text(df, input_col="default_response_0", output_col="default_response_1")
    notes = df["default_response_0"].astype(str)
    for w in _WRAPPERS:
        notes = notes.str.replace(w, "", regex=False).str.strip()
    notes = notes.tolist()
    gts = [_reanchor_dates(convert_to_xml(g), n, i, st)
           for i, (g, n) in enumerate(zip(df["default_response_1"], notes))]
    if verbose:
        print(f"{st.n_notes} notes ({path}) | tag repairs {st.tag_repairs} | gold dates {st.gold_dates}: "
              f"{st.dates_already_verbatim} verbatim, {st.dates_reanchored} re-anchored, "
              f"{len(st.dates_unanchored)} not found in note")
    return notes, gts, st


def load_tagged_gold(path: str = DEFAULT_PATH, gold_col: str = "Revision") -> list[str]:
    """The repaired, preprocessed inline-tagged gold (for scripts that derive the note
    from the gold itself, e.g. robustness_eval's perturbations)."""
    from edgedeid.data.preprocessing import preprocess_text

    df = pd.read_excel(path)
    if gold_col != "default_response_1":
        df = df.rename(columns={gold_col: "default_response_1"})
    gold = df["default_response_1"].astype(str)
    for pat, rep in _TAG_REPAIRS:
        gold = gold.str.replace(pat, rep, regex=True)
    df["default_response_1"] = gold
    with redirect_stdout(io.StringIO()):
        df = preprocess_text(df, input_col="default_response_0", output_col="default_response_1")
    return df["default_response_1"].astype(str).tolist()
