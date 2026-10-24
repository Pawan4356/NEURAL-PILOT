"""Loop-level regressions for the audit's Phase 0 findings (G1, G3, G7, G9-G11)."""

import numpy as np
import pandas as pd
import pytest

import joblib

from neuralpilot.graph import nodes
from neuralpilot import llm_client
from neuralpilot.agents.requirement_agent import InvalidClarification
from neuralpilot.orchestrator import apply_clarification, create_run, run_to_completion
from neuralpilot.repository import ExperimentRepository
from neuralpilot.schemas import (
    ConstraintWeights,
    DecisionOutput,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    RequirementSpec,
    TuningStrategy,
)


class ScriptedClient:
    """Returns scripted responses per schema and records every user prompt."""

    def __init__(self, responses_by_schema):
        self.responses_by_schema = responses_by_schema
        self.indices = {k: 0 for k in responses_by_schema}
        self.prompts: dict[type, list[str]] = {k: [] for k in responses_by_schema}

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        self.prompts.setdefault(schema, []).append(user_prompt)
        responses = self.responses_by_schema[schema]
        idx = min(self.indices[schema], len(responses) - 1)
        self.indices[schema] += 1
        return responses[idx]


@pytest.fixture
def repo(tmp_path):
    r = ExperimentRepository(db_path=tmp_path / "test.sqlite3")
    yield r
    r.close()


def _linear_data_csv(tmp_path, n=300):
    # y is a noisy linear function: logistic regression clearly beats an
    # unpruned decision tree on cross-validation.
    rng = np.random.default_rng(0)
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


def _reflections():
    return [
        ReflectionOutput(weakest_block="model_family", rationale="a"),
        ReflectionOutput(weakest_block="preprocessing", rationale="b"),
        ReflectionOutput(weakest_block="hyperparameters", rationale="c"),
    ]


def _run(tmp_path, repo, client, csv=None):
    llm_client._default_client = client
    state = create_run(csv or _linear_data_csv(tmp_path), "data.csv", "predict y")
    return run_to_completion(state, repo=repo)


def test_best_iteration_is_finalized_not_the_last(tmp_path, repo):
    """G1: a run that gets worse must still ship its best attempt."""
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("LinearModel"), _plan("DecisionTree")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [
                DecisionOutput(decision="replan", rationale="try something else"),
                DecisionOutput(decision="accept", rationale="done"),
            ],
        }
    )
    state = _run(tmp_path, repo, client)

    assert state.status == "finalized", state.error
    first, second = state.iteration_logs
    assert first.cv_mean > second.cv_mean  # the run got worse
    report = state.final_report
    assert report["best_iteration"] == 1
    assert report["model_family"] == "LinearModel"
    saved = joblib.load(state.model_path)
    assert type(saved.named_steps["model"]).__name__ == "LogisticRegression"
    assert report["composite_score"] == pytest.approx(first.composite_score)


def test_holdout_is_scored_exactly_once(tmp_path, repo, monkeypatch):
    """G7: attempts are compared on CV; the holdout is used once at the end."""
    calls = []
    real = nodes.holdout_evaluate

    def spy(pipeline, prepared):
        calls.append(1)
        return real(pipeline, prepared)

    monkeypatch.setattr(nodes, "holdout_evaluate", spy)
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("LinearModel"), _plan("DecisionTree"), _plan("RandomForest")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [
                DecisionOutput(decision="replan", rationale="r"),
                DecisionOutput(decision="replan", rationale="r"),
                DecisionOutput(decision="accept", rationale="a"),
            ],
        }
    )
    state = _run(tmp_path, repo, client)
    assert state.status == "finalized", state.error
    assert len(state.iteration_logs) == 3
    assert len(calls) == 1 and state.holdout_evaluations == 1
    assert state.final_report["holdout_metrics"] == state.final_metrics
    assert set(state.final_report["cv_metrics"]) >= {"f1_weighted", "accuracy"}


def test_replan_prompt_contains_attempt_history(tmp_path, repo):
    """G3: replan used to send a prompt byte-identical to iteration 1."""
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("LinearModel"), _plan("DecisionTree")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [
                DecisionOutput(decision="replan", rationale="r"),
                DecisionOutput(decision="accept", rationale="a"),
            ],
        }
    )
    state = _run(tmp_path, repo, client)
    first_prompt, second_prompt = client.prompts[PlanSpec]
    assert first_prompt != second_prompt
    assert "Already tried" not in first_prompt
    assert "Already tried" in second_prompt
    assert state.iteration_logs[0].signature in second_prompt
    assert "REPLAN" in second_prompt


def test_duplicate_plan_is_rejected_then_replaced_deterministically(tmp_path, repo):
    """G3: the planner keeps proposing the same plan; the orchestrator re-asks
    once with an explicit rejection, then substitutes an untried candidate."""
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("RandomForest")],  # always the same
            ReflectionOutput: _reflections(),
            DecisionOutput: [
                DecisionOutput(decision="refine", rationale="r"),
                DecisionOutput(decision="accept", rationale="a"),
            ],
        }
    )
    state = _run(tmp_path, repo, client)
    assert state.status == "finalized", state.error
    prompts = client.prompts[PlanSpec]
    assert len(prompts) == 3  # iteration 1, iteration 2, one re-ask
    assert "was already executed" in prompts[2]

    first, second = state.iteration_logs
    assert second.plan_source == "deterministic_fallback"
    assert second.signature != first.signature
    assert second.plan_summary["model_family"] != "RandomForest"
    assert any("deterministic fallback" in a for a in state.final_report["assumptions"])


def test_verification_failure_is_persisted_as_failed_run(tmp_path, repo):
    """G10: failed runs used to vanish."""
    path = tmp_path / "constant.csv"
    pd.DataFrame({"x": np.arange(50) * 0.5 % 7, "y": [1] * 50}).to_csv(path, index=False)
    client = ScriptedClient(
        {RequirementSpec: [_requirement()], PlanSpec: [_plan("LinearModel")]}
    )
    state = _run(tmp_path, repo, client, csv=path)

    assert state.status == "failed"
    assert "failed verification" in state.error
    [record] = repo.all_runs()
    assert record.run_id == state.run_id
    assert record.status == "failed"
    assert "failed verification" in record.error
    assert record.dataset_fingerprint == state.dataset_fingerprint
    assert record.lib_versions.get("scikit-learn")


def test_finalized_run_persists_decision_trace_and_versions(tmp_path, repo):
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("LinearModel")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [DecisionOutput(decision="accept", rationale="good enough")],
        }
    )
    state = _run(tmp_path, repo, client)
    record = repo.get_run(state.run_id)
    assert record.status == "finalized"
    assert record.iteration_logs[0]["decision_rationale"] == "good enough"
    assert record.iteration_logs[0]["decision_forced"] is False
    assert record.model_path == state.model_path
    assert record.final_metrics["holdout"] == state.final_metrics
    assert record.code_version
    assert state.final_report["reproducibility"]["dataset_fingerprint"] == state.dataset_fingerprint


def test_budget_exhaustion_forces_stop_without_llm_decision(tmp_path, repo):
    """G9: once the wall-clock budget is spent the run stops with its best attempt."""
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("LinearModel")],
            ReflectionOutput: _reflections(),
            # no DecisionOutput: calling the LLM decision agent would KeyError
        }
    )
    llm_client._default_client = client
    state = create_run(_linear_data_csv(tmp_path), "data.csv", "predict y")
    state.budget_s = 0.0
    state = run_to_completion(state, repo=repo)

    assert state.status == "finalized", state.error
    assert state.final_report["final_decision"] == "stop"
    assert state.final_report["final_decision_forced"] is True
    assert "budget" in state.final_report["final_decision_rationale"]


def test_cancel_before_any_attempt_fails_and_is_persisted(tmp_path, repo):
    client = ScriptedClient({RequirementSpec: [_requirement()]})
    llm_client._default_client = client
    state = create_run(_linear_data_csv(tmp_path), "data.csv", "predict y")
    state.cancel_event.set()
    state = run_to_completion(state, repo=repo)
    assert state.status == "failed"
    assert "cancelled" in state.error
    assert repo.get_run(state.run_id).status == "failed"


def test_invalid_clarification_is_rejected_and_problem_type_rederived(tmp_path, repo):
    """G11: the clarification answer is validated and the task re-derived."""
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement(target_column="guess", target_confidence=0.3, problem_type="regression")],
            PlanSpec: [_plan("LinearModel")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [DecisionOutput(decision="accept", rationale="a")],
        }
    )
    state = _run(tmp_path, repo, client)
    assert state.status == "awaiting_clarification"

    with pytest.raises(InvalidClarification) as excinfo:
        apply_clarification(state, "not_a_column")
    assert "x1" in str(excinfo.value)
    assert state.status == "awaiting_clarification"

    state = apply_clarification(state, "  Y ")
    assert state.requirement.target_column == "y"
    assert state.requirement.problem_type == "classification"
    assert any("re-derived" in a for a in state.requirement.assumptions)

    state = run_to_completion(state, repo=repo)
    assert state.status == "finalized", state.error


def test_unknown_target_from_llm_triggers_clarification(tmp_path, repo):
    client = ScriptedClient({RequirementSpec: [_requirement(target_column="Outcome", target_confidence=0.99)]})
    state = _run(tmp_path, repo, client)
    assert state.status == "awaiting_clarification"
    assert "couldn't find" in state.clarification_question


def test_target_column_case_is_normalised(tmp_path, repo):
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement(target_column="Y")],
            PlanSpec: [_plan("LinearModel")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [DecisionOutput(decision="accept", rationale="a")],
        }
    )
    state = _run(tmp_path, repo, client)
    assert state.status == "finalized", state.error
    assert state.final_report["target_column"] == "y"


def test_identifier_column_is_excluded_and_reported(tmp_path, repo):
    path = _linear_data_csv(tmp_path)
    df = pd.read_csv(path)
    df.insert(0, "CustomerID", np.arange(1, len(df) + 1))
    df.to_csv(path, index=False)
    client = ScriptedClient(
        {
            RequirementSpec: [_requirement()],
            PlanSpec: [_plan("LinearModel")],
            ReflectionOutput: _reflections(),
            DecisionOutput: [DecisionOutput(decision="accept", rationale="a")],
        }
    )
    state = _run(tmp_path, repo, client, csv=path)
    assert state.status == "finalized", state.error
    assert "CustomerID" in state.final_report["data"]["excluded_columns"]
    assert any("CustomerID" in a for a in state.final_report["assumptions"])
    assert "CustomerID" in client.prompts[PlanSpec][0]  # planner is told what was excluded
