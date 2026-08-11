"""Experiment repository (§5) — SQLite store + nearest-neighbor retrieval.

Retrieval is deliberately behind a single function,
`retrieve_similar_experiments(meta_features, k)`, so V2 can swap in
pgvector/embeddings without touching any agent-facing contract.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .meta_features import DatasetMetaFeatures

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset_name TEXT,
    problem_type TEXT,
    target_column TEXT,
    meta_features_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    weakest_block_history_json TEXT NOT NULL,
    final_decision TEXT,
    final_composite_score REAL
);
"""


@dataclass
class ExperimentRecord:
    run_id: str
    dataset_name: str
    problem_type: str
    target_column: str
    meta_features: dict
    plan: dict
    scores: list[float]
    weakest_block_history: list[str]
    final_decision: str
    final_composite_score: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(cls, **kwargs) -> "ExperimentRecord":
        kwargs.setdefault("run_id", str(uuid.uuid4()))
        return cls(**kwargs)

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "ExperimentRecord":
        return cls(
            run_id=row["run_id"],
            created_at=row["created_at"],
            dataset_name=row["dataset_name"],
            problem_type=row["problem_type"],
            target_column=row["target_column"],
            meta_features=json.loads(row["meta_features_json"]),
            plan=json.loads(row["plan_json"]),
            scores=json.loads(row["scores_json"]),
            weakest_block_history=json.loads(row["weakest_block_history_json"]),
            final_decision=row["final_decision"],
            final_composite_score=row["final_composite_score"],
        )


class ExperimentRepository:
    def __init__(self, db_path: Path | str = config.DB_PATH):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save_run(self, record: ExperimentRecord) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, created_at, dataset_name, problem_type, target_column,
                 meta_features_json, plan_json, scores_json, weakest_block_history_json,
                 final_decision, final_composite_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.created_at,
                record.dataset_name,
                record.problem_type,
                record.target_column,
                json.dumps(record.meta_features),
                json.dumps(record.plan),
                json.dumps(record.scores),
                json.dumps(record.weakest_block_history),
                record.final_decision,
                record.final_composite_score,
            ),
        )
        self._conn.commit()

    def all_runs(self) -> list[ExperimentRecord]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [ExperimentRecord._from_row(r) for r in rows]

    def retrieve_similar_experiments(
        self, meta_features: DatasetMetaFeatures, k: int = config.RETRIEVAL_TOP_K
    ) -> list[ExperimentRecord]:
        """Nearest-neighbor (cosine similarity) retrieval over the 5-dim
        meta-feature vector (§5). No embeddings, no vector DB.
        """
        query_vec = np.array(meta_features.vector(), dtype=float)
        query_norm = np.linalg.norm(query_vec)

        scored: list[tuple[float, ExperimentRecord]] = []
        for record in self.all_runs():
            candidate_vec = np.array(record.meta_features.get("vector") or [], dtype=float)
            if candidate_vec.shape != query_vec.shape:
                continue
            candidate_norm = np.linalg.norm(candidate_vec)
            if query_norm == 0 or candidate_norm == 0:
                similarity = 1.0 if query_norm == candidate_norm else 0.0
            else:
                similarity = float(np.dot(query_vec, candidate_vec) / (query_norm * candidate_norm))
            scored.append((similarity, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:k]]


def retrieve_similar_experiments(
    meta_features: DatasetMetaFeatures,
    k: int = config.RETRIEVAL_TOP_K,
    repo: ExperimentRepository | None = None,
) -> list[ExperimentRecord]:
    """Module-level convenience wrapper — the interface seam named in §5.

    Swapping in pgvector/embeddings later means changing this function's
    body only; every agent-facing call site stays the same.
    """
    owns_repo = repo is None
    repo = repo or ExperimentRepository()
    try:
        return repo.retrieve_similar_experiments(meta_features, k=k)
    finally:
        if owns_repo:
            repo.close()
