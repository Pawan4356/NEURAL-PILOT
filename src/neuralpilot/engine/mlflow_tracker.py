"""MLflow implementation of the `Tracker` seam (Phase 6, Deliverable 9.1).

Layout in the tracking store (`config.MLFLOW_TRACKING_URI`, SQLite):

    experiment "neuralpilot"
      parent run   one per NeuralPilot run: tags, holdout metrics, final model
        child run  one per attempt (candidate): params, CV metrics, artifacts

Every method is guarded: an MLflow error is logged and the run carries on,
because tracking is an observer of the experiment, not part of it.
The final model is logged with a schema only (column names and types),
never with an input example, so no rows are written to the tracking store.
"""

from __future__ import annotations

import functools
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from .. import config

logger = logging.getLogger(__name__)


def _guarded(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:  # tracking must never fail the run
            logger.warning("MLflow call %s failed; the run continues", method.__name__, exc_info=True)
            return None

    return wrapper


class MlflowTracker:
    def __init__(self, parent_run_id: str | None = None):
        import mlflow
        from mlflow.tracking import MlflowClient

        self._mlflow = mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        self._client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
        self._experiment_id = self._ensure_experiment()
        self.parent_run_id = parent_run_id
        self._attempt_run_id: str | None = None

    def _ensure_experiment(self) -> str:
        experiment = self._client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
        if experiment is not None:
            return experiment.experiment_id
        artifact_root = Path(config.MLFLOW_ARTIFACT_ROOT).resolve()
        return self._client.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=str(artifact_root))

    @_guarded
    def start_parent(self, run_id: str, tags: dict[str, str]) -> str | None:
        if self.parent_run_id is None:
            run = self._client.create_run(
                self._experiment_id,
                tags={**tags, "mlflow.runName": f"neuralpilot-{run_id[:8]}", "neuralpilot.run_id": run_id},
            )
            self.parent_run_id = run.info.run_id
        return self.parent_run_id

    @_guarded
    def start_attempt(self, signature: str, params: dict, tags: dict[str, str] | None = None) -> None:
        if self.parent_run_id is None:
            return
        run = self._client.create_run(
            self._experiment_id,
            tags={
                "mlflow.parentRunId": self.parent_run_id,
                "mlflow.runName": f"attempt-{signature}",
                "neuralpilot.signature": signature,
                **(tags or {}),
            },
        )
        self._attempt_run_id = run.info.run_id
        for key, value in params.items():
            self._client.log_param(self._attempt_run_id, key, _param_text(value))

    @_guarded
    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._attempt_run_id is None:
            return
        for key, value in metrics.items():
            if value is not None:
                self._client.log_metric(self._attempt_run_id, key, float(value))

    @_guarded
    def log_artifact(self, name: str, payload: Any) -> None:
        run_id = self._attempt_run_id or self.parent_run_id
        if run_id is None:
            return
        self._upload_json(run_id, name, payload)

    @_guarded
    def end_attempt(self, status: str) -> None:
        if self._attempt_run_id is None:
            return
        self._client.set_terminated(self._attempt_run_id, "FINISHED" if status == "ok" else "FAILED")
        self._attempt_run_id = None

    @_guarded
    def log_final(self, *, pipeline: Any, input_columns: dict[str, str], holdout: dict[str, float], report: dict) -> None:
        if self.parent_run_id is None:
            return
        for key, value in holdout.items():
            self._client.log_metric(self.parent_run_id, f"holdout_{key}", float(value))
        self._upload_json(self.parent_run_id, "final_report.json", report)
        self._upload_json(self.parent_run_id, "load_report.json", report.get("data", {}).get("load_report", {}))
        self._log_model(pipeline, input_columns)

    def _upload_json(self, run_id: str, name: str, payload: Any) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(json.dumps(payload, indent=2, default=str))
            self._client.log_artifact(run_id, str(path))

    def _log_model(self, pipeline: Any, input_columns: dict[str, str]) -> None:
        from mlflow.models import ModelSignature
        from mlflow.sklearn import log_model
        from mlflow.types import ColSpec, DataType, Schema

        dtypes = {"double": DataType.double, "string": DataType.string}
        schema = Schema([ColSpec(dtypes[kind], name) for name, kind in input_columns.items()])
        # cloudpickle, not MLflow's default skops format: skops refuses the tree
        # types inside tree ensembles. The store is local and written by us, so
        # pickle loading is acceptable here (and only here).
        with self._mlflow.start_run(run_id=self.parent_run_id):
            log_model(
                pipeline,
                name="model",
                signature=ModelSignature(inputs=schema),
                serialization_format="cloudpickle",
            )

    @_guarded
    def end(self, status: str) -> None:
        if self.parent_run_id is not None:
            self._client.set_terminated(self.parent_run_id, "FINISHED" if status == "finalized" else "FAILED")


def _param_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:500]  # MLflow limits parameter values to 6000 characters; keep them short
