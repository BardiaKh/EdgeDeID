"""QC v2 precision fixes. v1 behaviour must stay unchanged unless a switch is on."""
import pytest

from edgedeid.qc.detectors import detect
from edgedeid.qc.repair import repair_xml


def _dates(note, strict):
    return [d.text for d in detect(note, strict_dates=strict) if d.type == "date"]


def test_vitals_are_not_dates_under_strict():
    note = "ABG 7.41/27/162. BP 41/27/162. Seen on 03/22/2023 and 2024-03-22."
    assert "41/27/162" in _dates(note, strict=False)          # v1 behaviour kept
    assert _dates(note, strict=True) == ["03/22/2023", "2024-03-22"]


def test_lenient_repair_recovers_mismatched_tags():
    bad = "<phi_entities>\n<names/>\n<places>\n<place id=\"1\">CALVERT</device>\n</places>\n</phi_entities>"
    assert repair_xml(bad).status == "unrecoverable"
    r = repair_xml(bad, lenient=True)
    assert r.status == "repaired_lenient" and r.root is not None


def test_guarded_tier3():
    pytest.importorskip("spacy")
    from edgedeid.qc.detectors_contextual import detect_contextual
    try:
        detect_contextual("x")
    except ImportError:
        pytest.skip("en_core_web_sm missing")
    note = ("Patient seen at Massachusetts General Hospital by Dr. Rachel Lee. ECG and CT ordered. "
            "Follow up in 8 weeks and to contact the clinic. Lives at 789 Maple Street in Boston.")
    got = {(d.type, d.text) for d in detect_contextual(note, guarded=True)}
    assert ("place", "Massachusetts General Hospital") in got
    assert not any(t in {"ECG", "CT"} for _, t in got)
    assert not any("weeks and to contact" in t for _, t in got)
    assert ("address", "789 Maple Street") in got
    assert detect_contextual(note.upper(), guarded=True) == []    # all-caps note: skipped
