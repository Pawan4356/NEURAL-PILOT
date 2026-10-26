"""Curated documentation sources, read from `knowledge_sources/*.yaml` (Phase 8).

Each file lists pages with the registry components they explain and the task
types they apply to. Entries are checked when they are loaded: the host must
be on the fetch allow-list, every component id must exist in the registry, and
every task type must be known. A bad entry stops ingestion with a message that
names the file, rather than indexing a page nobody can use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..registry import list_components
from .fetch import host_allowed

TASK_TYPES = ("classification", "regression")
TRUST_LEVELS = ("official", "paper", "derived")


@dataclass(frozen=True)
class SourcePage:
    url: str
    title: str
    library: str
    component_ids: tuple[str, ...]
    task_types: tuple[str, ...]
    trust: str


def registry_component_names() -> set[str]:
    names: set[str] = set()
    for category in ("encoder", "scaler", "model_family", "tuner"):
        names.update(list_components(category))
    return names


def load_sources(directory: Path | str) -> list[SourcePage]:
    pages: list[SourcePage] = []
    for path in sorted(Path(directory).glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in document.get("pages", []):
            pages.append(_page(path, document.get("library", path.stem), entry))
    return pages


def _page(path: Path, library: str, entry: dict) -> SourcePage:
    where = f"{path.name}: {entry.get('url', '<no url>')}"
    url = str(entry.get("url", ""))
    if not host_allowed(url):
        raise ValueError(f"{where}: host is not on the documentation allow-list")
    components = tuple(entry.get("component_ids") or ())
    unknown = sorted(set(components) - registry_component_names())
    if unknown:
        raise ValueError(f"{where}: component ids not in the registry: {unknown}")
    tasks = tuple(entry.get("task_types") or TASK_TYPES)
    if not set(tasks) <= set(TASK_TYPES):
        raise ValueError(f"{where}: unknown task types {sorted(set(tasks) - set(TASK_TYPES))}")
    trust = entry.get("trust", "official")
    if trust not in TRUST_LEVELS:
        raise ValueError(f"{where}: trust must be one of {TRUST_LEVELS}")
    return SourcePage(
        url=url,
        title=str(entry.get("title", url)),
        library=library,
        component_ids=components,
        task_types=tasks,
        trust=trust,
    )
