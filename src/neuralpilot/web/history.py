"""Past runs as the UI sees them, read from the experiment store.

A run is held in memory only while it is active (clarifying, running, or
waiting to be cancelled). Once it finishes, its experiment row is the source of
truth: the report, iteration history and model path all come from there.
This is what lets a user reopen a run after a restart.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .. import config
from ..accounts import OwnedRun
from ..memory.store import ExperimentRecord
from ..repository import ExperimentRepository

FINISHED_STATUSES = ("finalized", "failed")
INTERRUPTED = "interrupted"  # owned by a user, but the server restarted before it finished


@contextmanager
def experiment_store() -> Iterator[ExperimentRepository]:
    repo = ExperimentRepository(db_path=config.DB_PATH)
    try:
        yield repo
    finally:
        repo.close()


def serialize_record(record: ExperimentRecord) -> dict:
    """The same shape `web.app._serialize` returns for a live run, for a finished one.
    Only `current_task` is empty: it describes work in progress, which a finished run does not have."""
    return {
        "run_id": record.run_id,
        "dataset_name": record.dataset_name,
        "status": record.status,
        "current_task": "",
        "activity_log": record.activity_log,
        "clarification_question": None,
        "iteration": len(record.iteration_logs),
        "max_iterations": _max_iterations_of(record),
        "iteration_logs": record.iteration_logs,
        "final_report": record.final_report,
        "error": record.error,
    }


def _max_iterations_of(record: ExperimentRecord) -> int:
    """The cap this run ran with. Runs saved before the cap was per-run did not
    record it, so they fall back to the server default."""
    repro = (record.final_report or {}).get("reproducibility") or {}
    return repro.get("max_iterations") or config.MAX_ITERATIONS


def interrupted_view(owned: OwnedRun) -> dict:
    """Detail view for a run the server lost before it finished. Nothing was stored,
    so the page gets the goal and a clear message, not an empty result."""
    return {
        "run_id": owned.run_id,
        "dataset_name": owned.dataset_name,
        "status": INTERRUPTED,
        "current_task": "",
        "activity_log": [],
        "clarification_question": None,
        "iteration": 0,
        "max_iterations": config.MAX_ITERATIONS,
        "iteration_logs": [],
        "final_report": None,
        "error": "The server restarted before this run finished, so its results were not saved.",
    }


def summarize(owned: OwnedRun, live_status: str | None, record: ExperimentRecord | None) -> dict:
    """One row for the 'your runs' list. Live status wins, then the stored status,
    and a run with neither is reported as interrupted."""
    if live_status is not None:
        status = live_status
    elif record is not None:
        status = record.status
    else:
        status = INTERRUPTED
    return {
        "run_id": owned.run_id,
        "dataset_name": owned.dataset_name,
        "goal": owned.goal,
        "created_at": owned.created_at,
        "status": status,
    }
