"""Experiment tracking seam (engine component 9).

The engine and graph report through the `Tracker` protocol. Two
implementations exist: `NoopTracker` (used when tracking is off or MLflow
is not installed) and `MlflowTracker` in `mlflow_tracker.py`. A tracking
failure must never fail a run, so implementations swallow and log their
own errors.

Relationship to SQLite (Deliverable 9.1): MLflow keeps the numeric record
of every attempt; SQLite keeps the application state and stores the MLflow
parent run id so the two can be joined.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from .. import config

logger = logging.getLogger(__name__)


class Tracker(Protocol):
    parent_run_id: str | None

    def start_parent(self, run_id: str, tags: dict[str, str]) -> str | None: ...

    def start_attempt(self, signature: str, params: dict, tags: dict[str, str] | None = None) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, name: str, payload: Any) -> None: ...

    def end_attempt(self, status: str) -> None: ...

    def log_final(self, *, pipeline: Any, input_columns: dict[str, str], holdout: dict[str, float], report: dict) -> None: ...

    def end(self, status: str) -> None: ...


class NoopTracker:
    parent_run_id: str | None = None

    def start_parent(self, run_id: str, tags: dict[str, str]) -> str | None:
        return None

    def start_attempt(self, signature: str, params: dict, tags: dict[str, str] | None = None) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, name: str, payload: Any) -> None:
        pass

    def end_attempt(self, status: str) -> None:
        pass

    def log_final(self, *, pipeline: Any, input_columns: dict[str, str], holdout: dict[str, float], report: dict) -> None:
        pass

    def end(self, status: str) -> None:
        pass


def make_tracker(parent_run_id: str | None = None) -> Tracker:
    """The tracker the configuration asks for. Falls back to no-op with a
    warning when MLflow is disabled or cannot be imported."""
    if not config.MLFLOW_ENABLED:
        return NoopTracker()
    try:
        from .mlflow_tracker import MlflowTracker
    except ImportError as exc:  # MLflow is an optional runtime dependency
        logger.warning("MLflow tracking disabled: %s", exc)
        return NoopTracker()
    return MlflowTracker(parent_run_id=parent_run_id)
