"""Few-shot example sets for the ``<examples>`` block of the prompt.

Each set is a JSON list of ``{"original": note, "anonymized": phi_xml}`` dicts,
the shape the SFT data's ``examples`` column used (``anonymized`` is the inner
XML, without the ``<phi_entities>`` wrapper).

``icu_nursing`` -- three invented ICU nursing notes written for the PhysioNet
deid (MIMIC) corpus' conventions: clinician and relative names tagged, ALL-CAPS
and lowercase single-token names, bare m/d dates and years, ages only above 89,
pagers as phones; shift times, vent settings, vitals and shorthand (MAE, PERRLA,
FOLEY, NEO, MICU) left untagged. No text is taken from the corpus.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from importlib import resources

_VERBATIM_TAGS = ("value", "age", "original", "place", "address", "phone",
                  "email", "id", "url", "device")


def load_examples(name: str = "icu_nursing") -> list[dict]:
    """Load a named example set and check that each one is self-consistent."""
    text = resources.files(__package__).joinpath(f"{name}.json").read_text()
    examples = json.loads(text)
    for i, ex in enumerate(examples):
        root = ET.fromstring(f"<phi_entities>{ex['anonymized']}</phi_entities>")
        for el in root.iter():
            if el.tag in _VERBATIM_TAGS and el.text and el.text not in ex["original"]:
                raise ValueError(f"{name}[{i}]: {el.text!r} is not in its note")
    return examples
