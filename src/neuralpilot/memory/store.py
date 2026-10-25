"""Experiment store: SQLite persistence and evidence-gated retrieval (Phase 7).

Every run that reaches a terminal state is stored, including failures, with
its iteration log, metrics, model path, dataset fingerprint and library
versions. Every measured attempt of a run is stored in its own table, so a
later run can warm-start from the parameters that actually worked.

Retrieval returns only *usable evidence* (see `is_usable_evidence`): finalized
runs, not flagged as suspect, whose plan still resolves against the current
registry, and whose final model met the user's hard constraints. Failed runs
are kept for the record but never returned as evidence.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from .. import config
from ..meta_features import DatasetMetaFeatures
from ..registry import component_exists
from ..schemas import AttemptSummary, UnsupportedRecommendation
from .migrations import migrate
from .similarity import memory_vector, similarity

EVIDENCE_OK = "ok"
EVIDENCE_SUSPECT_LEAKAGE = "suspect_leakage"


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
    evidence_quality: str = EVIDENCE_OK

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
            evidence_quality=row["evidence_quality"],
        )


class SimilarExperiment(NamedTuple):
    record: ExperimentRecord
    similarity: float  # exp(-distance) in (0, 1]; 1 = identical meta-features


def plan_components_registered(plan: dict) -> bool:
    """True when every component a stored plan names still exists in the
    registry. A renamed or removed component makes the stored configuration
    impossible to reproduce, so its evidence is not used (Phase 7 hard filter)."""
    family = plan.get("model_family")
    if family and not component_exists("model_family", family):
        return False
    tuning = plan.get("tuning") or {}
    if tuning.get("component") and not component_exists("tuner", tuning["component"]):
        return False
    for step in plan.get("preprocessing") or []:
        category = step.get("category")
        if category in ("encoder", "scaler") and not component_exists(category, step.get("component", "")):
            return False
    return True


def is_usable_evidence(record: ExperimentRecord) -> bool:
    """Whether a stored run may be cited as evidence for a new run."""
    if record.status != "finalized" or record.evidence_quality != EVIDENCE_OK:
        return False
    if record.final_report is not None and not record.final_report.get("feasible", True):
        return False  # the final model missed a hard constraint: not a model to copy
    return plan_components_registered(record.plan)


class ExperimentRepository:
    def __init__(self, db_path: Path | str = config.DB_PATH):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        migrate(self._conn)

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
                 lib_versions_json, final_report_json, mlflow_run_id, evidence_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record.evidence_quality,
            ),
        )
        self._conn.commit()

    def save_attempts(self, run_id: str, attempts: list[AttemptSummary]) -> None:
        """Replace this run's attempt rows. Saving twice leaves one row per signature."""
        self._conn.execute("DELETE FROM attempts WHERE run_id = ?", (run_id,))
        self._conn.executemany(
            """
            INSERT INTO attempts
                (run_id, signature, iteration, source, model_family, primary_metric, cv_mean, cv_std,
                 composite_score, feasible, latency_ms_per_1k, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    a.signature,
                    a.iteration,
                    a.source,
                    a.model_family,
                    a.primary_metric,
                    a.cv_mean,
                    a.cv_std,
                    a.composite_score,
                    int(a.feasible),
                    a.latency_ms_per_1k,
                    json.dumps(a.best_params, default=str),
                )
                for a in attempts
            ],
        )
        self._conn.commit()

    def attempts_for_run(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM attempts WHERE run_id = ? ORDER BY iteration, signature", (run_id,)
        ).fetchall()
        return [
            {
                "signature": r["signature"],
                "iteration": r["iteration"],
                "source": r["source"],
                "model_family": r["model_family"],
                "primary_metric": r["primary_metric"],
                "cv_mean": r["cv_mean"],
                "cv_std": r["cv_std"],
                "composite_score": r["composite_score"],
                "feasible": bool(r["feasible"]),
                "latency_ms_per_1k": r["latency_ms_per_1k"],
                "best_params": json.loads(r["params_json"]),
            }
            for r in rows
        ]

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
        """Nearest usable runs by `memory_vector` similarity. Only usable
        evidence of the same problem type is a candidate, and results below
        `min_similarity` are dropped rather than passed off as evidence."""
        query_vec = memory_vector(meta_features.to_dict(), problem_type)

        scored: list[SimilarExperiment] = []
        for record in self.all_runs(status="finalized"):
            if problem_type is not None and record.problem_type != problem_type:
                continue
            if not is_usable_evidence(record):
                continue
            score = similarity(query_vec, memory_vector(record.meta_features, record.problem_type))
            if score >= min_similarity:
                scored.append(SimilarExperiment(record, score))

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
