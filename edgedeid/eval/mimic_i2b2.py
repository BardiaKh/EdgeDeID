"""Convert i2b2/n2c2-style paired MIMIC de-identification files into ground
truth for EdgeDeID's evaluation pipeline.

The corpus ships as two aligned files, keyed by a shared record id in
``START_OF_RECORD=1||||<id>||||...||||END_OF_RECORD`` blocks:

- ``id.text``     — the original note, real PHI intact.
- ``id.res.txt``  — the same note with every PHI span replaced by an
                     ``[**Category label**]`` placeholder.

There is no separate annotation file, so ground truth is recovered by
diffing the two aligned records: wherever ``id.res.txt`` has a bracket
placeholder, the same span in ``id.text`` is the PHI surrogate text, and
the bracket's label gives its i2b2 category. Categories are mapped onto
EdgeDeID's schema (names/ages/dates/places/addresses/phones/emails/ids/
urls/devices); labels that don't map cleanly are left untagged and
counted, not guessed at.

Output reuses the project's existing flat-tag -> phi_entities conversion
(``edgedeid.data.xml_utils.convert_to_xml``), the same path already used
for the external English eval set (see notebooks/3_eval/en_eval.ipynb),
so the result plugs into the existing eval notebooks unchanged.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from edgedeid.data.xml_utils import convert_to_xml

_RECORD_RE = re.compile(
    r"START_OF_RECORD=(?P<doc_id>\d+)\|\|\|\|(?P<line_id>\d+)\|\|\|\|(?P<body>.*?)\|\|\|\|END_OF_RECORD",
    re.DOTALL,
)
_BRACKET_RE = re.compile(r"\[\*\*.*?\*\*\]")
_TOKEN_RE = re.compile(r"\S+|\s+")

# 4-digit numbers that look like a plausible calendar year.
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
# MM-DD, YYYY-MM-DD, or similar dash-separated numeric date fragments.
_DATEY_RE = re.compile(r"^\d{1,4}(-\d{1,2}){1,2}$")
# bare 1-2 digit number: i2b2 convention for ages < 90 (ages >= 90 are
# aggregated into the literal label "Age over 90" per HIPAA safe harbor).
_BARE_AGE_RE = re.compile(r"^\d{1,2}$")

# category classification rules, checked in order — first match wins.
# each entry: (regex over the label with any trailing " <instance-number>"
# stripped, our schema tag).
_LABEL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^age", re.I), "age"),
    (re.compile(r"date|month/year", re.I), "date"),
    (re.compile(r"hospital|location|ward\s*name|clinic|company|university", re.I), "place"),
    (re.compile(r"street address|po box|address", re.I), "address"),
    (re.compile(r"telephone|fax|pager", re.I), "phone"),
    (re.compile(r"e-?mail", re.I), "email"),
    (re.compile(r"url|web address|ip address", re.I), "url"),
    (re.compile(r"device|serial number|vehicle", re.I), "device"),
    (re.compile(r"unit number|medical record|ssn|social security|health plan|license|"
                r"account|provider number|job number|biometric", re.I), "id"),
    (re.compile(r"name|doctor|patient|initial", re.I), "name"),
]


def classify_label(raw_label: str) -> str | None:
    """Map an i2b2 bracket label (e.g. ``"Last Name (STitle)"``) to an
    EdgeDeID schema tag, or ``None`` if it doesn't map cleanly.
    """
    label = re.sub(r"\s+\d+$", "", raw_label).strip()
    if _DATEY_RE.match(label) or _YEAR_RE.match(label):
        return "date"
    for pattern, tag in _LABEL_RULES:
        if pattern.search(label):
            return tag
    if _BARE_AGE_RE.match(label):
        return "age"
    return None


def _infer_name_type(raw_label: str) -> str:
    low = raw_label.lower()
    if "doctor" in low:
        return "provider"
    if "patient" in low:
        return "patient"
    return "other"


def parse_records(path: str) -> dict[str, str]:
    """Parse a ``START_OF_RECORD/END_OF_RECORD`` file into {record_id: body}.

    The record id is the ``doc_id||||line_id`` pair — ``line_id`` alone is
    only unique *within* a document, not across the corpus (it resets for
    each new ``doc_id``).
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {
        f'{m.group("doc_id")}-{m.group("line_id")}': m.group("body")
        for m in _RECORD_RE.finditer(content)
    }


def _tokenize(text: str) -> list[str]:
    """Split into whitespace/non-whitespace tokens, treating each
    ``[**...**]`` placeholder as one atomic token so diff alignment can't
    split a bracket across tokens."""
    tokens: list[str] = []
    pos = 0
    for m in _BRACKET_RE.finditer(text):
        if m.start() > pos:
            tokens.extend(_TOKEN_RE.findall(text[pos:m.start()]))
        tokens.append(m.group())
        pos = m.end()
    if pos < len(text):
        tokens.extend(_TOKEN_RE.findall(text[pos:]))
    return tokens


_TRAILING_PUNCT = ";:,.)]}>-"
_LEADING_PUNCT = "([{<-"


def _trim_punct(span: str) -> tuple[str, int, int]:
    """Strip leading/trailing sentence punctuation picked up by the diff
    (e.g. a date immediately followed by ';' with no space) that a model
    would not be expected to reproduce. Returns (trimmed, left_trim, right_trim)."""
    trimmed = span
    left = 0
    while trimmed and trimmed[0] in _LEADING_PUNCT:
        trimmed = trimmed[1:]
        left += 1
    right = 0
    while trimmed and trimmed[-1] in _TRAILING_PUNCT:
        trimmed = trimmed[:-1]
        right += 1
    return trimmed, left, right


@dataclass
class PhiSpan:
    text: str
    label: str  # raw i2b2 bracket label
    tag: str | None  # mapped EdgeDeID schema tag, or None if unmapped


@dataclass
class ConvertedRecord:
    record_id: str
    source_note: str  # id.text body, unmodified — model input
    tagged_note: str  # id.text body with mapped spans wrapped in flat XML tags
    ground_truth_xml: str  # phi_entities XML (via convert_to_xml)
    spans: list[PhiSpan] = field(default_factory=list)
    unmapped_labels: list[str] = field(default_factory=list)
    ambiguous_multi_bracket: int = 0
    diff_without_bracket: int = 0


def _align_record(text_body: str, res_body: str) -> tuple[list[tuple[int, int, str]], int, int]:
    """Diff-align one record pair; return (spans, n_ambiguous, n_no_bracket).

    spans: list of (char_start, char_end, raw_i2b2_label) into text_body.
    """
    text_tokens = _tokenize(text_body)
    res_tokens = _tokenize(res_body)

    sm = difflib.SequenceMatcher(None, text_tokens, res_tokens, autojunk=False)
    spans: list[tuple[int, int, str]] = []
    n_ambiguous = 0
    n_no_bracket = 0

    # running char offset for text_tokens, indexed by token position
    offsets = [0] * (len(text_tokens) + 1)
    for i, t in enumerate(text_tokens):
        offsets[i + 1] = offsets[i] + len(t)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        res_span_tokens = res_tokens[j1:j2]
        brackets = [t for t in res_span_tokens if _BRACKET_RE.fullmatch(t)]
        if not brackets:
            n_no_bracket += 1
            continue

        span_start, span_end = offsets[i1], offsets[i2]
        raw_span = text_body[span_start:span_end]

        if len(brackets) == 1:
            stripped = raw_span.strip()
            if not stripped:
                continue
            lead = len(raw_span) - len(raw_span.lstrip())
            trimmed, ltrim, rtrim = _trim_punct(stripped)
            if not trimmed:
                continue
            start = span_start + lead + ltrim
            end = start + len(trimmed)
            label = brackets[0][3:-3]  # strip [** **]
            spans.append((start, end, label))
            continue

        # multiple PHI items with no separating "equal" token — split the
        # text span on whitespace and pair positionally if counts match.
        chunks = [c for c in re.split(r"(\s+)", raw_span) if c.strip()]
        if len(chunks) == len(brackets):
            cursor = span_start
            for chunk, bracket in zip(chunks, brackets):
                idx = text_body.index(chunk, cursor)
                trimmed, ltrim, rtrim = _trim_punct(chunk)
                cursor = idx + len(chunk)
                if trimmed:
                    spans.append((idx + ltrim, idx + ltrim + len(trimmed), bracket[3:-3]))
        else:
            # best-effort fallback: tag the whole span with the first label
            n_ambiguous += 1
            stripped = raw_span.strip()
            if stripped:
                lead = len(raw_span) - len(raw_span.lstrip())
                trimmed, ltrim, rtrim = _trim_punct(stripped)
                if trimmed:
                    start = span_start + lead + ltrim
                    spans.append((start, start + len(trimmed), brackets[0][3:-3]))

    return spans, n_ambiguous, n_no_bracket


def convert_record(record_id: str, text_body: str, res_body: str) -> ConvertedRecord:
    raw_spans, n_ambiguous, n_no_bracket = _align_record(text_body, res_body)

    phi_spans: list[PhiSpan] = []
    unmapped: list[str] = []
    for start, end, label in raw_spans:
        surrogate = text_body[start:end]
        tag = classify_label(label)
        phi_spans.append(PhiSpan(text=surrogate, label=label, tag=tag))
        if tag is None:
            unmapped.append(label)

    # build the flat inline-tagged text (highest offset first, so earlier
    # offsets stay valid while we insert).
    tagged = text_body
    name_counter = 0
    place_counter = 0
    id_counter = 0
    ordered = sorted(zip(raw_spans, phi_spans), key=lambda x: x[0][0], reverse=True)
    for (start, end, _label), span in ordered:
        if span.tag is None:
            continue
        content = tagged[start:end]
        if span.tag == "name":
            name_counter += 1
            open_tag = f'<name id="{name_counter}" type="{_infer_name_type(span.label)}">'
            close_tag = "</name>"
        elif span.tag == "place":
            place_counter += 1
            open_tag = f'<place id="{place_counter}">'
            close_tag = "</place>"
        elif span.tag == "id":
            id_counter += 1
            open_tag = f'<id id="{id_counter}">'
            close_tag = "</id>"
        else:
            open_tag, close_tag = f"<{span.tag}>", f"</{span.tag}>"
        tagged = tagged[:start] + open_tag + content + close_tag + tagged[end:]

    ground_truth_xml = convert_to_xml(tagged, sort_by_position=True)

    return ConvertedRecord(
        record_id=record_id,
        source_note=text_body,
        tagged_note=tagged,
        ground_truth_xml=ground_truth_xml,
        spans=phi_spans,
        unmapped_labels=unmapped,
        ambiguous_multi_bracket=n_ambiguous,
        diff_without_bracket=n_no_bracket,
    )


def convert_corpus(text_path: str, res_path: str) -> list[ConvertedRecord]:
    text_records = parse_records(text_path)
    res_records = parse_records(res_path)

    def _sort_key(rid: str) -> tuple[int, int]:
        doc, line = rid.split("-")
        return int(doc), int(line)

    common_ids = sorted((text_records.keys() & res_records.keys()), key=_sort_key)
    missing = (text_records.keys() ^ res_records.keys())
    if missing:
        import warnings
        warnings.warn(f"{len(missing)} record id(s) present in only one file, skipped: {sorted(missing)[:10]}...")

    return [
        convert_record(rid, text_records[rid], res_records[rid])
        for rid in common_ids
    ]


# ─── human gold standard (PhysioNet deid 1.1: id-phi.phrase) ──────────────────
#
# ``id.res.txt`` above is NOT a human annotation: it is the output of the
# ``deid.pl`` software itself (PhysioNet describes ``id.res`` as "the
# deidentified output text, with PHI replaced by tags"; 99.5% of its spans
# overlap deid.pl's own GSoutput/id.phi). PhysioNet reports deid.pl at recall
# 0.967 / PPV 0.748 against the human gold standard, so scoring against
# ``id.res.txt`` grades models against a noisy automatic tagger.
#
# The human gold standard ships as ``id-phi.phrase`` (one line per PHI:
# ``patient note start end type phrase``), 1,779 spans over the same 2,434
# notes. Its offsets are one character short of ``parse_records``' bodies
# (measured: all 1,779 phrases sit at exactly ``start + 1``); spans are placed
# by their phrase there, and any phrase that does not match exactly raises,
# rather than being silently mis-tagged.

GOLD_TYPE_TO_TAG: dict[str, str] = {
    "HCPName": "name",            # health-care provider
    "PTName": "name",
    "PTNameInitial": "name",
    "RelativeProxyName": "name",
    "Date": "date",
    "DateYear": "date",           # e.g. "MI '92" -- a year, not an age
    "Location": "place",          # hospitals, wards, towns: one gold type, mapped
                                  # to place as the id.res.txt conversion did
    "Phone": "phone",
    "Age": "age",                 # HIPAA: only ages >= 90 are PHI (4 spans, all "98")
    "Other": "id",                # 3 spans: "#rg17", "ref #8336652", "2115" after
                                  # "admission note" -- identifier-like numbers
}
_GOLD_NAME_TYPE = {"HCPName": "provider", "PTName": "patient", "PTNameInitial": "patient"}
GOLD_OFFSET_SHIFT = 1


def read_gold_phrases(phrase_path: str) -> dict[str, list[tuple[int, str, str]]]:
    """Parse ``id-phi.phrase`` into {record_id: [(start, gold_type, phrase), ...]}."""
    out: dict[str, list[tuple[int, str, str]]] = {}
    with open(phrase_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            patient, note, start, _end, gtype, phrase = line.rstrip("\n").split(None, 5)
            out.setdefault(f"{patient}-{note}", []).append((int(start), gtype, phrase.strip()))
    return out


def convert_record_gold(record_id: str, text_body: str,
                        gold: list[tuple[int, str, str]]) -> ConvertedRecord:
    spans: list[tuple[int, int, str]] = []
    for start, gtype, phrase in gold:
        s = start + GOLD_OFFSET_SHIFT
        e = s + len(phrase)
        if text_body[s:e] != phrase:
            raise ValueError(f"{record_id}: gold phrase {phrase!r} not found at offset {s} "
                             f"(text there: {text_body[s:e]!r})")
        if gtype not in GOLD_TYPE_TO_TAG:
            raise ValueError(f"{record_id}: unknown gold PHI type {gtype!r}")
        spans.append((s, e, gtype))
    spans.sort()
    # One overlap exists in the corpus (record 11-1: "Kessler-Adventist" and
    # "Adventist Hosp", both Location): same-type overlaps merge into their union;
    # a cross-type overlap has no single right tag and raises.
    merged: list[tuple[int, int, str]] = []
    for s, e, g in spans:
        if merged and s < merged[-1][1]:
            ps, pe, pg = merged[-1]
            if GOLD_TYPE_TO_TAG[pg] != GOLD_TYPE_TO_TAG[g]:
                raise ValueError(f"{record_id}: overlapping gold spans of different type at {ps}-{pe} and {s}-{e}")
            merged[-1] = (ps, max(pe, e), pg)
        else:
            merged.append((s, e, g))
    spans = merged

    tagged = text_body
    counters = {"name": 0, "place": 0, "id": 0}
    phi_spans = [PhiSpan(text=text_body[s:e], label=g, tag=GOLD_TYPE_TO_TAG[g]) for s, e, g in spans]
    for (s, e, gtype), span in sorted(zip(spans, phi_spans), key=lambda x: x[0][0], reverse=True):
        tag = span.tag
        if tag in counters:
            counters[tag] += 1
            attrs = f' id="{counters[tag]}"'
            if tag == "name":
                attrs += f' type="{_GOLD_NAME_TYPE.get(gtype, "other")}"'
            open_tag = f"<{tag}{attrs}>"
        else:
            open_tag = f"<{tag}>"
        tagged = tagged[:s] + open_tag + tagged[s:e] + f"</{tag}>" + tagged[e:]

    return ConvertedRecord(
        record_id=record_id,
        source_note=text_body,
        tagged_note=tagged,
        ground_truth_xml=convert_to_xml(tagged, sort_by_position=True),
        spans=phi_spans,
    )


def convert_corpus_gold(text_path: str, phrase_path: str) -> list[ConvertedRecord]:
    """Ground truth from the HUMAN gold standard, same record order as convert_corpus."""
    text_records = parse_records(text_path)
    gold = read_gold_phrases(phrase_path)
    unknown = gold.keys() - text_records.keys()
    if unknown:
        raise ValueError(f"gold names {len(unknown)} record(s) absent from {text_path}: {sorted(unknown)[:5]}")

    def _sort_key(rid: str) -> tuple[int, int]:
        doc, line = rid.split("-")
        return int(doc), int(line)

    return [convert_record_gold(rid, text_records[rid], gold.get(rid, []))
            for rid in sorted(text_records, key=_sort_key)]
