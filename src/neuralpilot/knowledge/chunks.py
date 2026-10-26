"""Passages for the documentation index (Phase 8, Deliverable 7.3).

A page is split on its headings into sections, and long sections are cut into
windows of about `MAX_WORDS` words with `OVERLAP_WORDS` shared between
neighbours, so an answer that straddles a boundary is still found whole. A
chunk's id is a hash of its text and source, so re-ingesting the same page
writes the same ids and changes nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

MAX_WORDS = 450  # roughly 600 tokens of English prose
OVERLAP_WORDS = 60


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:24]


def windows(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Word windows over `text`; a text that fits is returned whole."""
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    step = max_words - overlap
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), step)]


def chunk_sections(sections: list[Section], metadata: dict) -> list[Chunk]:
    """Chunks for one page. `metadata` (title, url, library, task_types, ...) is
    copied onto every chunk together with the section heading."""
    chunks: list[Chunk] = []
    url = str(metadata.get("url", ""))
    for section in sections:
        for piece in windows(section.text):
            chunk_meta = {**metadata, "section": section.heading}
            chunks.append(Chunk(id=content_hash(url, section.heading, piece), text=piece, metadata=chunk_meta))
    return chunks
