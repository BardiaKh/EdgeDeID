"""Abstract base class for all PHI de-identification baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaselineModel(ABC):
    """Unified interface for baseline models.

    Each subclass must implement `predict_entities`, which takes a raw medical
    note and returns a dict mapping PHI entity type → list of text strings.
    The types should match our schema keys: name, age, date, place, address,
    phone, email, id, url, device.
    """

    name: str = "baseline"

    # Schema keys this model is *capable* of emitting. Used by
    # pipeline.eval.compute_metrics to scope macro-F1 / per-entity reporting to
    # what the model can actually detect (entities outside this set show as N/A
    # instead of a misleading 0.0). Defaults to the full schema.
    supported_entities: set[str] = {
        "name", "age", "date", "place", "address",
        "phone", "email", "id", "url", "device",
    }

    @abstractmethod
    def predict_entities(self, note: str) -> dict[str, list[str]]:
        """Return {entity_type: [text, …]} for all PHI detected in note."""

    def predict_xml(self, note: str) -> str:
        """Convert predict_entities output to a minimal <phi_entities> XML string.

        This lets baselines be evaluated with the same pipeline.eval.compute_metrics
        pipeline used for our models.
        """
        entities = self.predict_entities(note)
        lines = ["<phi_entities>"]
        tag_to_parent = {
            "name": "names", "age": "ages", "date": "dates", "place": "places",
            "address": "addresses", "phone": "phones", "email": "emails",
            "id": "ids", "url": "urls", "device": "devices",
        }
        for singular, parent in tag_to_parent.items():
            values = entities.get(singular, [])
            if values:
                lines.append(f"  <{parent}>")
                for v in values:
                    if singular == "name":
                        lines.append(f"    <name><variations><value>{v}</value></variations></name>")
                    elif singular == "date":
                        lines.append(f"    <date><original>{v}</original><formatted>{v}</formatted></date>")
                    else:
                        lines.append(f"    <{singular}>{v}</{singular}>")
                lines.append(f"  </{parent}>")
            else:
                lines.append(f"  <{parent}/>")
        lines.append("</phi_entities>")
        return "\n".join(lines)
