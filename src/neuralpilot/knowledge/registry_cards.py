"""Registry cards: one passage per registry component, generated from the
registry itself (Phase 8, Deliverable 7.3). They are rebuilt on every ingest,
so they cannot disagree with the code that will run the component.
"""

from __future__ import annotations

from ..registry import REGISTRY
from .chunks import Chunk, content_hash

CARD_SOURCE_TYPE = "registry_card"


def registry_cards() -> list[Chunk]:
    chunks = []
    for category in sorted(REGISTRY):
        for name in sorted(REGISTRY[category]):
            chunks.append(_card(category, REGISTRY[category][name]))
    return chunks


def _card(category: str, spec) -> Chunk:
    url = f"registry://{category}/{spec.name}"
    aliases = ", ".join(spec.aliases) or "none"
    alternatives = ", ".join(spec.alternatives) or "none"
    lines = [
        f"{spec.name} is a {category.replace('_', ' ')} in the Neural Pilot registry.",
        f"Supported problem types: {', '.join(spec.supports)}.",
        f"Other names the planner may use: {aliases}.",
        f"Semantic group: {spec.semantic_group or 'none'}. Ranked alternatives: {alternatives}.",
        f"Resource class: {spec.resource_class}.",
    ]
    if spec.interp_class:
        lines.append(f"Interpretability class: {spec.interp_class}.")
    if spec.max_rows_hint:
        lines.append(f"Over budget above {spec.max_rows_hint} training rows.")
    text = " ".join(lines)
    metadata = {
        "source_type": CARD_SOURCE_TYPE,
        "library": "neuralpilot",
        "title": f"{spec.name} (registry card)",
        "url": url,
        "section": category,
        "component_ids": [spec.name],
        "task_types": list(spec.supports),
        "trust": "derived",
    }
    return Chunk(id=content_hash(url, category, text), text=text, metadata=metadata)
