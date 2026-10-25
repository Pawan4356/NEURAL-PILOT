import pytest

from neuralpilot.agents.planning_agent import generate_plan
from neuralpilot.agents.reflection_agent import reflect
from neuralpilot.agents.requirement_agent import (
    apply_clarification_answer,
    needs_clarification,
    understand_requirement,
)
from neuralpilot.meta_features import extract_meta_features
from neuralpilot.schemas import (
    BlockDiagnostics,
    ConstraintWeights,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    RequirementSpec,
    TuningStrategy,
    ValidationReport,
)

import pandas as pd


class FakeClient:
    """Stub LLMClient that returns a preset object and records calls."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        self.calls.append((system_prompt, user_prompt, schema))
        return self.response


def _df():
    return pd.DataFrame({"x1": range(50), "x2": ["a", "b"] * 25, "y": [0, 1] * 25})


# --- Requirement Understanding Agent ---


def test_understand_requirement_applies_default_weight_fallback():
    low_conf_response = RequirementSpec(
        target_column="y",
        target_confidence=0.95,
        problem_type="classification",
        constraint_weights=ConstraintWeights(accuracy=0.5, latency=0.3, interpretability=0.2),
        weight_confidence=0.4,  # below threshold
    )
    client = FakeClient(low_conf_response)
    meta = extract_meta_features(_df(), target_column="y")

    spec = understand_requirement("predict y", meta, client=client)

    assert spec.constraint_weights.accuracy == pytest.approx(0.7)
    assert spec.constraint_weights.latency == pytest.approx(0.15)
    assert spec.constraint_weights.interpretability == pytest.approx(0.15)
    assert any("default constraint weights" in a for a in spec.assumptions)


def test_understand_requirement_keeps_explicit_weights_when_confident():
    high_conf_response = RequirementSpec(
        target_column="y",
        target_confidence=0.95,
        problem_type="classification",
        constraint_weights=ConstraintWeights(accuracy=0.3, latency=0.1, interpretability=0.6),
        weight_confidence=0.85,
    )
    client = FakeClient(high_conf_response)
    meta = extract_meta_features(_df(), target_column="y")

    spec = understand_requirement("predict y, must be explainable", meta, client=client)
    assert spec.constraint_weights.interpretability == pytest.approx(0.6)
    assert spec.assumptions == []


def test_needs_clarification_threshold():
    low = RequirementSpec(
        target_column="y", target_confidence=0.5, problem_type="classification",
        constraint_weights=ConstraintWeights(), weight_confidence=0.9,
    )
    high = low.model_copy(update={"target_confidence": 0.8})
    assert needs_clarification(low)
    assert not needs_clarification(high)


def test_apply_clarification_answer_overwrites_target():
    spec = RequirementSpec(
        target_column="maybe_target", target_confidence=0.4, problem_type="classification",
        constraint_weights=ConstraintWeights(), weight_confidence=0.9,
    )
    updated = apply_clarification_answer(spec, "definitely_target")
    assert updated.target_column == "definitely_target"
    assert updated.target_confidence == 1.0
    assert any("confirmed by user" in a for a in updated.assumptions)


# --- Planning Agent ---


def test_generate_plan_passes_through_and_sets_iteration():
    plan_response = PlanSpec(
        preprocessing=[PreprocessingStep(category="encoder", component="OneHotEncoder")],
        model_family="RandomForest",
        tuning=TuningStrategy(),
    )
    client = FakeClient(plan_response)
    requirement = RequirementSpec(
        target_column="y", target_confidence=0.9, problem_type="classification",
        constraint_weights=ConstraintWeights(), weight_confidence=0.9,
    )
    meta = extract_meta_features(_df(), target_column="y")

    plan = generate_plan(requirement, meta, [], client=client, iteration=3)
    assert plan.iteration == 3
    assert len(client.calls) == 1


def test_generate_plan_refine_mentions_weakest_block_in_prompt():
    plan_response = PlanSpec(
        preprocessing=[PreprocessingStep(category="encoder", component="OneHotEncoder")],
        model_family="RandomForest",
        tuning=TuningStrategy(),
    )
    client = FakeClient(plan_response)
    requirement = RequirementSpec(
        target_column="y", target_confidence=0.9, problem_type="classification",
        constraint_weights=ConstraintWeights(), weight_confidence=0.9,
    )
    meta = extract_meta_features(_df(), target_column="y")
    previous = plan_response
    reflection = ReflectionOutput(weakest_block="hyperparameters", rationale="undertuned")

    generate_plan(
        requirement, meta, [], client=client, iteration=2, previous_plan=previous, reflection=reflection
    )
    _, user_prompt, _ = client.calls[0]
    assert "hyperparameters" in user_prompt
    assert "refine iteration" in user_prompt


# --- Reflection Agent ---


def test_reflect_returns_weakest_block():
    response = ReflectionOutput(weakest_block="model_family", rationale="underfitting")
    client = FakeClient(response)
    report = ValidationReport(
        composite_score=0.5, accuracy_norm=0.5, latency_norm=0.2, interp_norm=0.6,
        raw_metrics={"accuracy": 0.5}, per_block_diagnostics=[BlockDiagnostics(block="model_family", duration_seconds=1.0)],
    )
    result = reflect(report, client=client)
    assert result.weakest_block == "model_family"
