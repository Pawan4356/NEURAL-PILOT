"""Per-invocation dependencies handed to graph nodes.

LangGraph passes this object to every node as `runtime.context`. It holds
what cannot (or must not) be checkpointed: the database connection, the
cancel flag, the live progress callback and an optional injected LLM
client. A resumed run simply gets a fresh `RunContext`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from ..engine.tracking import NoopTracker, Tracker
from ..llm_client import LLMClient
from ..repository import ExperimentRepository

logger = logging.getLogger(__name__)


def _ignore_progress(task: str, tag: str) -> None:
    pass


@dataclass
class RunContext:
    repo: ExperimentRepository
    cancel_event: threading.Event = field(default_factory=threading.Event)
    on_progress: Callable[[str, str], None] = _ignore_progress
    llm: LLMClient | None = None  # None = the process-wide default client
    run_id: str = ""
    tracker: Tracker = field(default_factory=NoopTracker)  # MLflow or no-op (Phase 6)

    def report(self, task: str, tag: str = "agent") -> None:
        """Tell the UI what the run is doing right now."""
        logger.info("run=%s task=%s", self.run_id, task)
        self.on_progress(task, tag)

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()
