"""Phase 4: a model family wins only when the measurements say so (Deliverable 8.7).

The planner here always proposes LinearModel. The data is an interaction
(the label depends on the product of two features), which no linear model
can express. The run must measure the challengers, and the winner must be
one of them. The report must show that the other candidates were evaluated.
"""

import numpy as np
import pandas as pd
import pytest

from neuralpilot.orchestrator import create_run, run_to_completion
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

TREE_LIKE = {"DecisionTree", "RandomForest", "ExtraTrees", "HistGradientBoosting", "GradientBoosting", "XGBoost", "CatBoost"}


class ScriptedClient:
    def __init__(self, responses_by_schema):
        self.responses_by_schema = responses_by_schema
        self.indices = {k: 0 for k in responses_by_schema}

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        responses = self.responses_by_schema[schema]
        idx = min(self.indices[schema], len(responses) - 1)
        self.indices[schema] += 1
        return responses[idx]


def _interaction_csv(tmp_path, seed):
    rng = np.random.default_rng(seed)
    n = 1_500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = ((x1 * x2 > 0) ^ (rng.random(n) < 0.05)).astype(int)
    path = tmp_path / f"interaction_{seed}.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "y": y}).to_csv(path, index=False)
    return path


def _run(path, tmp_path):
    plan = PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family="LinearModel",
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )
    requirement = RequirementSpec(
        target_column="y",
        target_confidence=0.95,
        problem_type="classification",
        constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
        weight_confidence=0.95,
    )
    client = ScriptedClient(
        {
            RequirementSpec: [requirement],
            PlanSpec: [plan],
            ReflectionOutput: [ReflectionOutput(weakest_block="model_family", rationale="r")],
            DecisionOutput: [DecisionOutput(decision="accept", rationale="ok")],
        }
    )
    repo = ExperimentRepository(db_path=tmp_path / "exp.sqlite3")
    try:
        return run_to_completion(create_run(path, path.name, "predict y"), repo=repo, llm=client)
    finally:
        repo.close()


@pytest.mark.parametrize("seed", [0, 1])
def test_a_tree_model_wins_on_an_interaction_and_the_linear_model_was_still_measured(tmp_path, seed):
    state = _run(_interaction_csv(tmp_path, seed), tmp_path)
    assert state.status == "finalized", state.error

    evaluated = [c for c in state.iteration_logs[0].candidates if c["status"] == "ok"]
    families = {c["model_family"] for c in evaluated}
    assert "LinearModel" in families
    assert len(evaluated) >= 3
    assert state.final_report["model_family"] in TREE_LIKE
    assert state.final_report["model_family"] != "LinearModel"


def test_the_report_lists_every_candidate_with_its_source(tmp_path):
    state = _run(_interaction_csv(tmp_path, 0), tmp_path)
    sources = {c["source"] for c in state.iteration_logs[0].candidates}
    assert {"planner", "baseline"} <= sources
    assert state.final_report["run_history"][0]["signature"] == state.iteration_logs[0].signature
