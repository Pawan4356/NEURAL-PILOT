"""Experiment repository (§5) — SQLite store + nearest-neighbor retrieval.

Every run that reaches a terminal state is stored, including failures,
with its iteration log (decisions, rationale, whether a guardrail forced
the decision), metrics, model path, dataset fingerprint and library
versions. Schema changes are applied as numbered migrations tracked by
`PRAGMA user_version`, so older databases upgrade in place.

Retrieval is behind `retrieve_similar_experiments(meta_features, k)`:
scaled meta-feature distance, filtered to the same task and to finalized
runs, with the similarity score returned alongside each record.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import config
from .meta_features import DatasetMetaFeatures, memory_vector
from .schemas import UnsupportedRecommendation

_BASE_SCHEMA = """
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

# Each entry upgrades the schema by one version. Never edit a released
# migration; append a new one.
_MIGRATIONS: list[list[str]] = [
    [
        "ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'finalized'",
        "ALTER TABLE runs ADD COLUMN error TEXT",
        "ALTER TABLE runs ADD COLUMN iteration_logs_json TEXT NOT NULL DEFAULT '[]'",
        "ALTER TABLE runs ADD COLUMN final_metrics_json TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE runs ADD COLUMN model_path TEXT",
        "ALTER TABLE runs ADD COLUMN dataset_fingerprint TEXT",
        "ALTER TABLE runs ADD COLUMN code_version TEXT",
        "ALTER TABLE runs ADD COLUMN lib_versions_json TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE runs ADD COLUMN final_report_json TEXT",
    ],
    [  # Phase 4 + 6: MLflow parent run, and components the registry could not use.
        "ALTER TABLE runs ADD COLUMN mlflow_run_id TEXT",
        """
        CREATE TABLE IF NOT EXISTS unsupported_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            requested TEXT NOT NULL,
            category TEXT NOT NULL,
            source TEXT NOT NULL,
            reason TEXT NOT NULL,
            alternative_used TEXT,
            semantic_change INTEGER NOT NULL,
            context_json TEXT NOT NULL
        )
        """,
    ],
]
SCHEMA_VERSION = len(_MIGRATIONS)


@dataclass
class ExperimentRecord:
    run_id: str
    dataset_name: str
    problem_type: str | None
    target_column: str | None
    meta_features: dict
    plan: dict
    scores: list[float]
    weakest_block_history: list[str]
    final_decision: str | None
    final_composite_score: float | None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "finalized"
    error: str | None = None
    iteration_logs: list[dict] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)
    model_path: str | None = None
    dataset_fingerprint: str | None = None
    code_version: str | None = None
    lib_versions: dict = field(default_factory=dict)
    final_report: dict | None = None
    mlflow_run_id: str | None = None

    @classmethod
    def new(cls, **kwargs) -> "ExperimentRecord":
        kwargs.setdefault("run_id", str(uuid.uuid4()))
        return cls(**kwargs)

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "ExperimentRecord":
        report = row["final_report_json"]
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
            status=row["status"],
            error=row["error"],
            iteration_logs=json.loads(row["iteration_logs_json"]),
            final_metrics=json.loads(row["final_metrics_json"]),
            model_path=row["model_path"],
            dataset_fingerprint=row["dataset_fingerprint"],
            code_version=row["code_version"],
            lib_versions=json.loads(row["lib_versions_json"]),
            final_report=json.loads(report) if report else None,
            mlflow_run_id=row["mlflow_run_id"],
        )


class SimilarExperiment(NamedTuple):
    record: ExperimentRecord
    similarity: float  # exp(-distance) in (0, 1]; 1 = identical meta-features


class ExperimentRepository:
    def __init__(self, db_path: Path | str = config.DB_PATH):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_BASE_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for target_version in range(version + 1, SCHEMA_VERSION + 1):
            for statement in _MIGRATIONS[target_version - 1]:
                self._conn.execute(statement)
            self._conn.execute(f"PRAGMA user_version = {target_version}")

    def close(self) -> None:
        self._conn.close()

    def save_run(self, record: ExperimentRecord) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, created_at, dataset_name, problem_type, target_column,
                 meta_features_json, plan_json, scores_json, weakest_block_history_json,
                 final_decision, final_composite_score, status, error, iteration_logs_json,
                 final_metrics_json, model_path, dataset_fingerprint, code_version,
                 lib_versions_json, final_report_json, mlflow_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.created_at,
                record.dataset_name,
                record.problem_type,
                record.target_column,
                json.dumps(record.meta_features, default=str),
                json.dumps(record.plan, default=str),
                json.dumps(record.scores),
                json.dumps(record.weakest_block_history),
                record.final_decision,
                record.final_composite_score,
                record.status,
                record.error,
                json.dumps(record.iteration_logs, default=str),
                json.dumps(record.final_metrics, default=str),
                record.model_path,
                record.dataset_fingerprint,
                record.code_version,
                json.dumps(record.lib_versions),
                json.dumps(record.final_report, default=str) if record.final_report is not None else None,
                record.mlflow_run_id,
            ),
        )
        self._conn.commit()

    def save_unsupported(self, run_id: str, rows: list[UnsupportedRecommendation]) -> None:
        """Replace this run's unsupported-recommendation rows (called once, at the end)."""
        self._conn.execute("DELETE FROM unsupported_recommendations WHERE run_id = ?", (run_id,))
        self._conn.executemany(
            """
            INSERT INTO unsupported_recommendations
                (run_id, iteration, requested, category, source, reason, alternative_used,
                 semantic_change, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row.iteration,
                    row.requested,
                    row.category,
                    row.source,
                    row.reason,
                    row.alternative_used,
                    int(row.semantic_change),
                    json.dumps(row.context, default=str),
                )
                for row in rows
            ],
        )
        self._conn.commit()

    def unsupported_for_run(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM unsupported_recommendations WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [
            {
                "iteration": r["iteration"],
                "requested": r["requested"],
                "category": r["category"],
                "source": r["source"],
                "reason": r["reason"],
                "alternative_used": r["alternative_used"],
                "semantic_change": bool(r["semantic_change"]),
                "context": json.loads(r["context_json"]),
            }
            for r in rows
        ]

    def all_runs(self, status: str | None = None) -> list[ExperimentRecord]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        return [ExperimentRecord._from_row(r) for r in rows]

    def get_run(self, run_id: str) -> ExperimentRecord | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return ExperimentRecord._from_row(row) if row else None

    def retrieve_similar_experiments(
        self,
        meta_features: DatasetMetaFeatures,
        k: int = config.RETRIEVAL_TOP_K,
        *,
        problem_type: str | None = None,
        min_similarity: float = config.MIN_SIMILARITY,
    ) -> list[SimilarExperiment]:
        """Nearest neighbours by Euclidean distance over the scaled
        memory_vector(); similarity = exp(-distance). Only finalized runs of
        the same problem type are candidates, and results below
        `min_similarity` are dropped rather than passed off as evidence.
        """
        query_vec = np.array(memory_vector(meta_features.to_dict(), problem_type), dtype=float)

        scored: list[SimilarExperiment] = []
        for record in self.all_runs(status="finalized"):
            if problem_type is not None and record.problem_type != problem_type:
                continue
            candidate_vec = np.array(memory_vector(record.meta_features, record.problem_type), dtype=float)
            similarity = math.exp(-float(np.linalg.norm(query_vec - candidate_vec)))
            if similarity >= min_similarity:
                scored.append(SimilarExperiment(record, similarity))

        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:k]


def retrieve_similar_experiments(
    meta_features: DatasetMetaFeatures,
    k: int = config.RETRIEVAL_TOP_K,
    repo: ExperimentRepository | None = None,
    *,
    problem_type: str | None = None,
    min_similarity: float = config.MIN_SIMILARITY,
) -> list[SimilarExperiment]:
    """Module-level convenience wrapper — the interface seam named in §5."""
    owns_repo = repo is None
    repo = repo or ExperimentRepository()
    try:
        return repo.retrieve_similar_experiments(
            meta_features, k=k, problem_type=problem_type, min_similarity=min_similarity
        )
    finally:
        if owns_repo:
            repo.close()
