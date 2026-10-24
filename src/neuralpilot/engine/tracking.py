"""Experiment tracking seam (engine component 9).

The engine reports every attempt through the `Tracker` protocol. Only the
no-op tracker exists today; an MLflow implementation slots in later
without touching the engine. Tracking failures must never fail a run.
"""

from __future__ import annotations

from typing import Protocol


class Tracker(Protocol):
    def start_attempt(self, signature: str, params: dict) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, name: str, payload: dict) -> None: ...

    def end_attempt(self, status: str) -> None: ...


class NoopTracker:
    def start_attempt(self, signature: str, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, name: str, payload: dict) -> None:
        pass

    def end_attempt(self, status: str) -> None:
        pass
