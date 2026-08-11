import numpy as np
import pandas as pd
import pytest

from agentic_automl import llm_client
from agentic_automl.orchestrator import apply_clarification, create_run, run_to_completion
from agentic_automl.repository import ExperimentRepository
from agentic_automl.schemas import (
    ConstraintWeights,
    DecisionOutput,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    RequirementSpec,
    TuningStrategy,
)


class ScriptedClient:
    def __init__(self, responses_by_schema: dict[type, list]):
        self.responses_by_schema = responses_by_schema
        self.indices = {k: 0 for k in responses_by_schema}
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        self.calls.append(schema)
        responses = self.responses_by_schema[schema]
        idx = min(self.indices[schema], len(responses) - 1)
        self.indices[schema] += 1
        return responses[idx]


def _dataset_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.choice(["a", "b", "c"], size=n),
            "y": rng.choice([0, 1], size=n),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _plan():
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family="RandomForest",
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )


@pytest.fixture
def repo(tmp_path):
    r = ExperimentRepository(db_path=tmp_path / "test.sqlite3")
    yield r
    r.close()


@pytest.fixture(autouse=True)
def reset_default_client():
    yield
    llm_client._default_client = None


def test_full_loop_finalizes_after_accept(tmp_path, repo, monkeypatch):
    monkeypatch.setattr(
        "agentic_automl.orchestrator.MODELS_DIR", tmp_path / "models", raising=False
    )
    import agentic_automl.orchestrator as orch

    orch.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scripted = ScriptedClient(
        {
            RequirementSpec: [
                RequirementSpec(
                    target_column="y",
                    target_confidence=0.95,
                    problem_type="classification",
                    # latency weight zeroed so composite_score is driven purely by
                    # accuracy_norm — identical plan+data+seed gives an identical
                    # score across iterations, isolating this test from real
                    # wall-clock latency jitter (which would otherwise sometimes
                    # trip the plateau guardrail's "no net gain" condition).
                    constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
                    weight_confidence=0.95,
                )
            ],
            PlanSpec: [_plan()],
            # Two distinct weakest_block values so the plateau guardrail's
            # "same weakest_block twice in a row" condition never matches —
            # a real refine would target a different block each time anyway.
            ReflectionOutput: [
                ReflectionOutput(weakest_block="hyperparameters", rationale="undertuned"),
                ReflectionOutput(weakest_block="model_family", rationale="different weak point"),
            ],
            DecisionOutput: [
                DecisionOutput(decision="refine", rationale="try again"),
                DecisionOutput(decision="accept", rationale="good enough"),
            ],
        }
    )
    llm_client._default_client = scripted

    dataset_path = _dataset_csv(tmp_path)
    state = create_run(dataset_path, "data.csv", "predict y")
    state = run_to_completion(state, repo=repo)

    assert state.status == "finalized"
    assert state.iteration == 2
    assert len(state.iteration_logs) == 2
    assert state.final_report is not None
    assert state.final_report["final_decision"] == "accept"
    assert len(repo.all_runs()) == 1


def test_clarification_pauses_then_resumes(tmp_path, repo, monkeypatch):
    import agentic_automl.orchestrator as orch

    monkeypatch.setattr(orch, "MODELS_DIR", tmp_path / "models", raising=False)
    orch.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scripted = ScriptedClient(
        {
            RequirementSpec: [
                RequirementSpec(
                    target_column="guess",
                    target_confidence=0.3,  # below threshold -> clarification
                    problem_type="classification",
                    constraint_weights=ConstraintWeights(),
                    weight_confidence=0.9,
                )
            ],
            PlanSpec: [_plan()],
            ReflectionOutput: [ReflectionOutput(weakest_block="model_family", rationale="weak model")],
            DecisionOutput: [DecisionOutput(decision="accept", rationale="good enough")],
        }
    )
    llm_client._default_client = scripted

    dataset_path = _dataset_csv(tmp_path)
    state = create_run(dataset_path, "data.csv", "predict something")
    state = run_to_completion(state, repo=repo)

    assert state.status == "awaiting_clarification"
    assert "y" in state.clarification_question or "Available columns" in state.clarification_question

    state = apply_clarification(state, "y")
    state = run_to_completion(state, repo=repo)

    assert state.status == "finalized"
    assert state.requirement.target_column == "y"
    assert any("confirmed by user" in a for a in state.final_report["assumptions"])
