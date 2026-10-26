"""Offline documentation ingestion (Phase 8, Deliverable 7.3).

    python -m neuralpilot.knowledge.ingest [--sources DIR] [--index DIR]

Fetches each curated page, splits it into passages, rebuilds the registry
cards, and writes everything to the index. Run it by hand when the sources or
the registry change. The request path only reads the index, so ingestion never
slows a run down and never depends on the network during one.

Re-ingesting is safe: chunk ids are content hashes, so unchanged pages write
the same rows, and the registry cards are replaced rather than appended to.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from .chunks import chunk_sections
from .fetch import fetch_html, html_sections
from .registry_cards import CARD_SOURCE_TYPE, registry_cards
from .sources import load_sources
from .store import KnowledgeStore


@dataclass
class IngestReport:
    pages: int = 0
    chunks: int = 0
    cards: int = 0
    failed: list[str] = field(default_factory=list)
    index_size: int = 0


def ingest(
    store: KnowledgeStore,
    sources_dir: Path | str,
    *,
    fetch: Callable[[str], str] = fetch_html,
) -> IngestReport:
    report = IngestReport()
    for page in load_sources(sources_dir):
        try:
            html = fetch(page.url)
        except Exception as exc:  # one unreachable page must not stop the rest
            report.failed.append(f"{page.url}: {exc}")
            continue
        metadata = {
            "source_type": "doc",
            "library": page.library,
            "title": page.title,
            "url": page.url,
            "component_ids": list(page.component_ids),
            "task_types": list(page.task_types),
            "trust": page.trust,
        }
        report.chunks += store.upsert(chunk_sections(html_sections(html), metadata))
        report.pages += 1

    store.delete_source_type(CARD_SOURCE_TYPE)
    report.cards = store.upsert(registry_cards())
    report.index_size = store.count()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the documentation index for the planner.")
    parser.add_argument("--sources", default=str(config.KNOWLEDGE_SOURCES_DIR), help="directory of *.yaml sources")
    parser.add_argument("--index", default=str(Path(config.KNOWLEDGE_DIR) / "chroma"), help="index directory")
    args = parser.parse_args(argv)

    store = KnowledgeStore(args.index)
    report = ingest(store, args.sources)
    print(
        f"pages={report.pages} passages={report.chunks} registry_cards={report.cards} "
        f"index_size={report.index_size}"
    )
    for failure in report.failed:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
