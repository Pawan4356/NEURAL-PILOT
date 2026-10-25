"""Retrieval evaluation for the documentation index (Phase 8, Deliverable 7.6).

Each labelled query names the page(s) that answer it. A query is a hit at
rank r when the r-th passage returned comes from one of those pages. We report
recall@5 (the share of queries with a hit in the top five) and MRR (the mean of
1/rank). Pages, not individual chunks, are the labels: a chunk id changes when
its wording does, but the page a question belongs to does not.

    python -m neuralpilot.knowledge.evaluate [--queries FILE] [--index DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import config
from .retrieve import retrieve
from .sources import registry_component_names
from .store import KnowledgeStore

RECALL_K = 5
MRR_DEPTH = 20


@dataclass(frozen=True)
class LabelledQuery:
    text: str
    expected_urls: tuple[str, ...]
    problem_type: str | None = None


@dataclass(frozen=True)
class EvalResult:
    recall_at_k: float
    mrr: float
    n_queries: int
    misses: tuple[str, ...]


def load_queries(path: Path | str) -> list[LabelledQuery]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [
        LabelledQuery(
            text=str(item["query"]),
            expected_urls=tuple(item["expected_urls"]),
            problem_type=item.get("problem_type"),
        )
        for item in document.get("queries", [])
    ]


def evaluate(store: KnowledgeStore, queries: list[LabelledQuery]) -> EvalResult:
    components = registry_component_names()
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    misses: list[str] = []
    for query in queries:
        passages = retrieve(
            store,
            query.text,
            problem_type=query.problem_type,
            registry_components=components,
            k=MRR_DEPTH,
            fetch=MRR_DEPTH * 2,
        )
        rank = next((i + 1 for i, p in enumerate(passages) if p.url in query.expected_urls), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank is not None and rank <= RECALL_K:
            hits_at_k += 1
        else:
            misses.append(query.text)
    n = len(queries)
    return EvalResult(
        recall_at_k=hits_at_k / n if n else 0.0,
        mrr=sum(reciprocal_ranks) / n if n else 0.0,
        n_queries=n,
        misses=tuple(misses),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the documentation index on labelled queries.")
    parser.add_argument("--queries", default="tests/rag_eval/queries.yaml")
    parser.add_argument("--index", default=str(Path(config.KNOWLEDGE_DIR) / "chroma"))
    args = parser.parse_args(argv)

    store = KnowledgeStore(args.index)
    result = evaluate(store, load_queries(args.queries))
    print(f"queries={result.n_queries} recall@{RECALL_K}={result.recall_at_k:.3f} mrr={result.mrr:.3f}")
    for miss in result.misses:
        print(f"  miss: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
