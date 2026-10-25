"""Documentation index on Chroma (Phase 8, Deliverable 7.2).

A thin wrapper so the rest of the code does not depend on chromadb's API:
upsert by chunk id (re-ingesting a page replaces its chunks, so the index
never holds two copies), delete by source type (registry cards are rebuilt
whole), and query with the result flattened into plain `Hit`s.

Chroma is imported here and only here, and the index is built only when a
caller asks for it, so the rest of the application does not pay for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from .chunks import Chunk

DOCS_COLLECTION = "docs"


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


def flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma stores scalar metadata only, so lists become comma-separated strings."""
    flat: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            flat[key] = ",".join(sorted(str(v) for v in value))
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


class KnowledgeStore:
    def __init__(self, path: Path | str, *, collection: str = DOCS_COLLECTION, embedding_function=None):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, **kwargs
        )

    def count(self) -> int:
        return int(self._collection.count())

    def upsert(self, chunks: list[Chunk]) -> int:
        """Store chunks under their content ids. Returns how many were written."""
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[flatten_metadata(c.metadata) for c in chunks],
        )
        return len(chunks)

    def delete_source_type(self, source_type: str) -> None:
        self._collection.delete(where={"source_type": source_type})

    def query(self, text: str, n_results: int) -> list[Hit]:
        """The `n_results` nearest chunks to `text`, nearest first."""
        available = self.count()
        if available == 0 or n_results <= 0:
            return []
        result = self._collection.query(query_texts=[text], n_results=min(n_results, available))
        return [
            Hit(id=chunk_id, text=document, metadata=dict(metadata or {}), distance=float(distance))
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def open_index(path: Path | str | None = None) -> KnowledgeStore | None:
    """The built index, or None when there is none (or RAG is switched off).
    Nothing is created here: a missing index means no passages, not an empty one."""
    index_path = Path(path) if path is not None else Path(config.KNOWLEDGE_DIR) / "chroma"
    if not config.RAG_ENABLED or not index_path.exists():
        return None
    return KnowledgeStore(index_path)
