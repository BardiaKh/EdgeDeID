"""Structural repair of the model's <phi_entities> XML block (Tier-2 QC).

A small model frequently emits *almost* valid XML: a missing closing tag, a
stray ``&``, spaces inside tags. These are deterministically recoverable and
must NOT be discarded (discarding = silent recall loss). Truly unrecoverable
output is reported so the orchestrator can escalate / fall back to a
detector-only recall floor.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# schema parent → child (full PHI schema, ordered as in the system prompt)
SCHEMA: tuple[tuple[str, str], ...] = (
    ("names", "name"), ("ages", "age"), ("dates", "date"),
    ("places", "place"), ("addresses", "address"), ("phones", "phone"),
    ("emails", "email"), ("ids", "id"), ("urls", "url"), ("devices", "device"),
)
_ALL_TAGS = {p for p, _ in SCHEMA} | {c for _, c in SCHEMA} | {
    "phi_entities", "variations", "value", "original", "formatted",
}

_BLOCK_RE = re.compile(
    r"<phi_entities\b[^>]*>.*?</phi_entities>", re.DOTALL | re.IGNORECASE
)

# Whitespace slips the model makes inside tags: "< name>", "</ name>", "< /name>".
# Both fixers are restricted to known schema tag names so that stray "<" in
# free-text entity values is never rewritten.
_TAG_ALT = "|".join(sorted(_ALL_TAGS, key=len, reverse=True))
_OPEN_SPACE_RE = re.compile(rf"<\s+({_TAG_ALT})\b", re.IGNORECASE)
_CLOSE_SPACE_RE = re.compile(rf"<\s*/\s*({_TAG_ALT})\s*>", re.IGNORECASE)


def fix_space_in_tags(text: str) -> str:
    """Drop whitespace between '<' and a known opening tag name ("< name>")."""
    return _OPEN_SPACE_RE.sub(r"<\1", text)


def fix_space_in_closing_tags(text: str) -> str:
    """Normalise whitespace in known closing tags ("</ name>", "< /name >")."""
    return _CLOSE_SPACE_RE.sub(r"</\1>", text)


@dataclass(slots=True)
class RepairResult:
    xml: str | None          # repaired, parseable <phi_entities> string (or None)
    status: str              # "ok" | "repaired" | "unrecoverable"
    root: ET.Element | None  # parsed root when xml is not None


def _try_parse(xml_str: str) -> ET.Element | None:
    try:
        return ET.fromstring(xml_str)
    except ET.ParseError:
        return None


def _escape_stray_ampersands(text: str) -> str:
    # & not starting a valid entity reference -> &amp;
    return re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", text)


def _close_dangling_tags(text: str) -> str:
    """Append missing closing tags for known schema tags, in LIFO order."""
    stack: list[str] = []
    for m in re.finditer(r"<(/?)([A-Za-z_][\w]*)\b[^>]*?(/?)>", text):
        closing, name, selfclose = m.group(1), m.group(2), m.group(3)
        if name not in _ALL_TAGS or selfclose:
            continue
        if closing:
            if name in stack:
                # pop down to the matching open tag
                while stack and stack[-1] != name:
                    stack.pop()
                if stack:
                    stack.pop()
        else:
            stack.append(name)
    for name in reversed(stack):
        text += f"</{name}>"
    return text


def repair_xml(model_output: str) -> RepairResult:
    """Extract and (if needed) deterministically repair the <phi_entities> block."""
    if not isinstance(model_output, str) or "<phi_entities" not in model_output:
        return RepairResult(None, "unrecoverable", None)

    m = _BLOCK_RE.search(model_output)
    if m:
        block = m.group(0)
        root = _try_parse(block)
        if root is not None:
            return RepairResult(block, "ok", root)
        candidate = block
    else:
        # opening tag present but no closer at all — take from first opening on
        start = model_output.lower().find("<phi_entities")
        candidate = model_output[start:]

    repaired = candidate
    for fix in (fix_space_in_closing_tags, fix_space_in_tags,
                _escape_stray_ampersands):
        repaired = fix(repaired)
    if "</phi_entities>" not in repaired:
        repaired = repaired.rstrip()
        repaired = _close_dangling_tags(repaired)
        if "</phi_entities>" not in repaired:
            repaired += "</phi_entities>"
    else:
        repaired = _close_dangling_tags(repaired)

    m2 = _BLOCK_RE.search(repaired)
    repaired = m2.group(0) if m2 else repaired

    root = _try_parse(repaired)
    if root is not None:
        return RepairResult(repaired, "repaired", root)

    return RepairResult(None, "unrecoverable", None)


def empty_phi_xml() -> str:
    """The canonical empty <phi_entities> skeleton (all parents self-closed)."""
    inner = "\n".join(f"<{p}/>" for p, _ in SCHEMA)
    return f"<phi_entities>\n{inner}\n</phi_entities>"
