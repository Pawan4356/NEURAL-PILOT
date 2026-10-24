"""Phase 3: whole-graph scenarios with a scripted LLM and tiny data.

The basic loop scenarios (accept at 1, refine -> accept, replan -> new
plan, verification failure x3, clarification pause/resume, budget, cancel)
live in tests/test_orchestrator*.py and now run on the graph. These cover
what is specific to LangGraph: the iteration cap, re-asking inside the
graph, the recursion limit, checkpoint contents and crash recovery.
"""

import numpy as np
import pandas as pd
import pytest

from neuralpilot import orchestrator
from neuralpilot.graph import compiled_graph, nodes
from neuralpilot.orchestrator import create_run, run_to_completion
from neuralpilot.repository import ExperimentRepository
from neuralpilot.schemas import (
    BestAttempt,
    ConstraintWeights,
    DecisionOutput,
    IterationLog,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    RequirementSpec,
    TuningStrategy,
)


class ScriptedClient:
    def __init__(self, responses_by_schema):
        self.responses_by_schema = responses_by_schema
        self.indices = {k: 0 for k in responses_by_schema}
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        self.calls.append(schema)
        responses = self.responses_by_schema[schema]
        idx = min(self.indices[schema], len(responses) - 1)
        self.indices[schema] += 1
        return responses[idx]


class ExplodingClient:
    def complete_json(self, *args, **kwargs):
        raise AssertionError("no LLM call expected")


class Crash(BaseException):
    """Simulates the process dying mid-run (not an ordinary node error)."""


@pytest.fixture
def repo(tmp_path):
    r = ExperimentRepository(db_path=tmp_path / "exp.sqlite3")
    yield r
    r.close()


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + x2 + rng.normal(scale=0.7, size=n) > 0).astype(int)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "y": y}).to_csv(path, index=False)
    return path


def _requirement(**overrides):
    fields = dict(
        target_column="y",
        target_confidence=0.95,
        problem_type="classification",
        constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
        weight_confidence=0.95,
    )
    fields.update(overrides)
    return RequirementSpec(**fields)


def _plan(model):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family=model,
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )


def _client(decisions=("accept",), overrides=None):
    responses = {
        RequirementSpec: [_requirement()],
        PlanSpec: [_plan("LinearModel"), _plan("DecisionTree"), _plan("RandomForest")],
        ReflectionOutput: [
            ReflectionOutput(weakest_block="model_family", rationale="a"),
            ReflectionOutput(weakest_block="preprocessing", rationale="b"),
            ReflectionOutput(weakest_block="hyperparameters", rationale="c"),
        ],
        DecisionOutput: [DecisionOutput(decision=d, rationale=d) for d in decisions],
    }
    responses.update(overrides or {})
    return ScriptedClient(responses)


def _checkpoint(run_id):
    return compiled_graph().get_state({"configurable": {"thread_id": run_id}})


def test_iteration_cap_finalizes_best_so_far(csv, repo):
    state = create_run(csv, "data.csv", "predict y")
    state.max_iterations = 2
    state = run_to_completion(state, repo=repo, llm=_client(decisions=("refine",)))

    assert state.status == "finalized", state.error
    assert len(state.iteration_logs) == 2
    report = state.final_report
    assert report["final_decision"] == "stop" and report["final_decision_forced"] is True
    assert "max_iterations" in report["final_decision_rationale"]
    assert report["best_iteration"] == 1  # LinearModel beats the unpruned tree


def test_invalid_answer_inside_the_graph_is_reasked_then_fails(csv, repo):
    """The façade validates answers first; this covers a resume that skips it."""
    client = _client(overrides={RequirementSpec: [_requirement(target_column="guess", target_confidence=0.2)]})
    state = run_to_completion(create_run(csv, "data.csv", "predict"), repo=repo, llm=client)
    assert state.status == "awaiting_clarification"

    state.pending_answer = "nope"
    state = run_to_completion(state, repo=repo, llm=client)
    assert state.status == "awaiting_clarification"
    assert "'nope' is not a column" in state.clarification_question

    state.pending_answer = "still_nope"
    state = run_to_completion(state, repo=repo, llm=client)
    assert state.status == "failed"
    assert "Could not determine the target column" in state.error
    assert repo.get_run(state.run_id).status == "failed"


def test_recursion_limit_stops_a_runaway_graph_and_persists_failure(csv, repo, monkeypatch):
    monkeypatch.setattr(orchestrator, "recursion_limit", lambda max_iterations: 4)
    state = run_to_completion(create_run(csv, "data.csv", "predict y"), repo=repo, llm=_client())
    assert state.status == "failed"
    assert "Recursion limit" in state.error
    assert repo.get_run(state.run_id).status == "failed"


def test_checkpoint_holds_typed_models_and_no_dataframes(csv, repo):
    state = run_to_completion(create_run(csv, "data.csv", "predict y"), repo=repo, llm=_client())
    assert state.status == "finalized", state.error

    values = _checkpoint(state.run_id).values  # read back from SQLite
    assert isinstance(values["best"], BestAttempt)
    assert isinstance(values["requirement"], RequirementSpec)
    assert all(isinstance(log, IterationLog) for log in values["iteration_logs"])
    assert not any(isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)) for v in values.values())

    # A finished run is not re-executed when asked again.
    state = run_to_completion(state, repo=repo, llm=ExplodingClient())
    assert state.status == "finalized"


def test_resume_after_crash_does_not_rerun_completed_attempts(csv, repo, monkeypatch):
    executed = []
    real_run_attempt = nodes.run_attempt
    monkeypatch.setattr(nodes, "run_attempt", lambda *a, **kw: executed.append(1) or real_run_attempt(*a, **kw))

    crashes = []
    real_reflect = nodes.reflection_agent

    def reflect_crashing_once(*args, **kwargs):
        if not crashes:
            crashes.append(1)
            raise Crash()
        return real_reflect(*args, **kwargs)

    monkeypatch.setattr(nodes, "reflection_agent", reflect_crashing_once)
    client = _client()
    state = create_run(csv, "data.csv", "predict y")
    with pytest.raises(Crash):
        run_to_completion(state, repo=repo, llm=client)
    # The planner's plan and its challengers were all measured before the crash.
    measured_before_crash = len(executed)
    assert measured_before_crash >= 1
    assert _checkpoint(state.run_id).next == ("reflect",)

    # As after a restart: the fitted pipeline is gone from memory, only the file remains.
    assert nodes.resources._pipelines.pop(state.run_id, None) is not None
    state = run_to_completion(state, repo=repo, llm=client)
    assert state.status == "finalized", state.error
    assert len(executed) == measured_before_crash  # no candidate was trained again
    assert state.holdout_evaluations == 1
