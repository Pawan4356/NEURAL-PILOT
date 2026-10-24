"""Phase 6: MLflow tracking. Every test uses its own tracking store under
tmp_path, so nothing is written to runs/.

Layout asserted here: one parent run per NeuralPilot run (tags, holdout
metrics, final model), one child run per candidate attempt (params, CV
metrics, plan artifact), and a saved model that reloads to the same
predictions as the joblib bundle.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from neuralpilot import config
from neuralpilot.engine.mlflow_tracker import MlflowTracker
from neuralpilot.engine.tracking import NoopTracker, make_tracker
from neuralpilot.orchestrator import create_run, run_to_completion
from neuralpilot.repository import ExperimentRepository
from neuralpilot.schemas import ConstraintWeights, DecisionOutput, PlanSpec, PreprocessingStep, ReflectionOutput, RequirementSpec, TuningStrategy

import joblib
import mlflow.sklearn  # noqa: F401  (attribute access below needs the submodule loaded)


class ScriptedClient:
    def __init__(self, responses_by_schema):
        self.responses_by_schema = responses_by_schema
        self.indices = {k: 0 for k in responses_by_schema}

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        responses = self.responses_by_schema[schema]
        idx = min(self.indices[schema], len(responses) - 1)
        self.indices[schema] += 1
        return responses[idx]


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Turn tracking on, with a store of its own."""
    monkeypatch.setattr(config, "MLFLOW_ENABLED", True)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)


@pytest.fixture
def repo(tmp_path):
    r = ExperimentRepository(db_path=tmp_path / "exp.sqlite3")
    yield r
    r.close()


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + x2 + rng.normal(scale=0.7, size=n) > 0).astype(int)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "y": y}).to_csv(path, index=False)
    return path


def _plan(model):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family=model,
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )


def _client():
    return ScriptedClient(
        {
            RequirementSpec: [
                RequirementSpec(
                    target_column="y",
                    target_confidence=0.95,
                    problem_type="classification",
                    constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
                    weight_confidence=0.95,
                )
            ],
            PlanSpec: [_plan("LinearModel")],
            ReflectionOutput: [ReflectionOutput(weakest_block="model_family", rationale="r")],
            DecisionOutput: [DecisionOutput(decision="accept", rationale="ok")],
        }
    )


def test_tracking_is_off_by_default_and_the_factory_says_so():
    assert isinstance(make_tracker(), NoopTracker)


def test_one_parent_and_one_child_per_candidate(csv, repo, mlflow_store):
    state = run_to_completion(create_run(csv, "data.csv", "predict y"), repo=repo, llm=_client())
    assert state.status == "finalized", state.error

    parent_id = repo.get_run(state.run_id).mlflow_run_id
    assert parent_id is not None
    parent = mlflow_store.get_run(parent_id)
    assert parent.data.tags["neuralpilot.run_id"] == state.run_id
    assert parent.data.tags["neuralpilot.problem_type"] == "classification"
    assert parent.info.status == "FINISHED"
    assert "holdout_f1_weighted" in parent.data.metrics

    experiment_id = mlflow_store.get_experiment_by_name(config.MLFLOW_EXPERIMENT).experiment_id
    children = mlflow_store.search_runs(
        [experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'"
    )
    ok_candidates = [c for c in state.iteration_logs[0].candidates if c["status"] == "ok"]
    assert len(children) == len(ok_candidates) >= 2
    for child in children:
        assert child.info.status == "FINISHED"
        assert "model_family" in child.data.params
        assert "cv_f1_weighted" in child.data.metrics
        assert "neuralpilot.candidate_source" in child.data.tags
    # The model itself is a MLflow logged model (checked by reloading it below).
    artifacts = {a.path for a in mlflow_store.list_artifacts(parent_id)}
    assert {"final_report.json", "load_report.json"} <= artifacts


def test_saved_model_reloads_to_the_same_predictions(csv, repo, mlflow_store):
    state = run_to_completion(create_run(csv, "data.csv", "predict y"), repo=repo, llm=_client())
    assert state.status == "finalized", state.error
    parent_id = repo.get_run(state.run_id).mlflow_run_id

    tracked = mlflow.sklearn.load_model(f"runs:/{parent_id}/model")
    bundle = joblib.load(state.model_path)
    features = pd.read_csv(csv)[["x1", "x2"]]
    np.testing.assert_array_equal(tracked.predict(features), bundle.predict(features))


def test_tracking_failure_never_fails_the_run(csv, repo, mlflow_store, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(MlflowClient, "create_run", broken)
    monkeypatch.setattr(MlflowClient, "log_metric", broken)
    state = run_to_completion(create_run(csv, "data.csv", "predict y"), repo=repo, llm=_client())
    assert state.status == "finalized", state.error
    assert state.final_report["mlflow_run_id"] is None
    assert repo.get_run(state.run_id).status == "finalized"


def test_factory_returns_the_mlflow_tracker_when_enabled(mlflow_store):
    tracker = make_tracker()
    assert isinstance(tracker, MlflowTracker)
    assert tracker.start_parent("abc", {"neuralpilot.dataset": "d"}) is not None
