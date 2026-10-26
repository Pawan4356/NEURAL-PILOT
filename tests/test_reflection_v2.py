"""Phase 9: Reflection v2 and its guards (Deliverable 9.5)."""

import pytest

from neuralpilot.agents.reflection_agent import reflect
from neuralpilot.engine import plan_signature
from neuralpilot.plan_resolution import PlanProfile, resolve
from neuralpilot.reflection_guard import split_violated_claims, suggestion_plan, usable_suggestions
from neuralpilot.schemas import (
    AttemptSummary,
    BlockDiagnostics,
    ConstraintResult,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    ReflectionSuggestion,
    TuningStrategy,
    ValidationReport,
)


class ScriptedClient:
    """Returns the scripted replies in order and records every prompt."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete_json(self, system, user, schema, **kwargs):
        self.prompts.append(user)
        return self.replies.pop(0)


def _report(*results: ConstraintResult) -> ValidationReport:
    return ValidationReport(
        composite_score=0.5,
        accuracy_norm=0.5,
        latency_norm=0.2,
        interp_norm=0.6,
        raw_metrics={"f1_weighted": 0.5},
        per_block_diagnostics=[BlockDiagnostics(block="model_family", duration_seconds=1.0)],
        constraint_results=list(results),
        feasible=all(r.passed for r in results if r.kind == "hard"),
    )


def _check(name, passed, kind="hard") -> ConstraintResult:
    return ConstraintResult(
        name=name, kind=kind, threshold="x", measured=1.0, passed=passed, explanation=f"{name} explained"
    )


def _plan(family="LinearModel", **overrides) -> PlanSpec:
    fields = dict(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family=family,
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )
    return PlanSpec(**{**fields, **overrides})


def _profile() -> PlanProfile:
    return PlanProfile(problem_type="classification", n_rows=1_000, categorical_cardinality=5)


def _reply(**overrides) -> ReflectionOutput:
    return ReflectionOutput(**{"weakest_block": "model_family", "rationale": "r", **overrides})


def _attempt(family) -> AttemptSummary:
    base = _plan(family)
    return AttemptSummary(
        signature=plan_signature(base, categorical_cardinality=5),
        iteration=1,
        model_family=family,
        preprocessing=["OneHotEncoder", "StandardScaler"],
        feature_engineering=[],
        tuning="NoTuning(0)",
        primary_metric="f1_weighted",
        cv_mean=0.7,
        cv_std=0.01,
        composite_score=0.6,
    )


# --- schema: old replies still parse ------------------------------------------


def test_a_reply_with_only_the_original_fields_still_parses():
    parsed = ReflectionOutput.model_validate({"weakest_block": "preprocessing", "rationale": "scaling"})
    assert parsed.proposed_action == "none"
    assert parsed.violated_constraints == [] and parsed.suggestions == []


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        ReflectionOutput(weakest_block="model_family", rationale="r", confidence=1.5)


# --- constraint claims ----------------------------------------------------------


def test_claims_are_split_into_failing_and_not_failing_names():
    assert split_violated_claims(["latency", "min_primary_metric"], ["latency"]) == (["latency"], ["min_primary_metric"])


def test_a_claim_that_matches_a_failing_constraint_is_kept():
    client = ScriptedClient(_reply(violated_constraints=["max_latency_ms_per_1k"]))
    report = _report(_check("max_latency_ms_per_1k", False), _check("min_primary_metric", True))
    result = reflect(report, client=client)
    assert result.violated_constraints == ["max_latency_ms_per_1k"]
    assert result.guard_notes == []
    assert len(client.prompts) == 1


def test_a_mismatched_claim_is_sent_back_once_and_the_retry_is_used():
    client = ScriptedClient(
        _reply(violated_constraints=["min_primary_metric"]),  # passed, so it is not a violation
        _reply(violated_constraints=["max_latency_ms_per_1k"]),
    )
    report = _report(_check("max_latency_ms_per_1k", False), _check("min_primary_metric", True))
    result = reflect(report, client=client)
    assert result.violated_constraints == ["max_latency_ms_per_1k"]
    assert len(client.prompts) == 2
    assert "Answer again" in client.prompts[1]
    assert any("asked again" in note for note in result.guard_notes)


def test_a_claim_still_mismatched_after_the_retry_is_dropped_and_noted():
    client = ScriptedClient(
        _reply(violated_constraints=["made_up"]),
        _reply(violated_constraints=["made_up", "max_latency_ms_per_1k"]),
    )
    report = _report(_check("max_latency_ms_per_1k", False))
    result = reflect(report, client=client)
    assert result.violated_constraints == ["max_latency_ms_per_1k"]
    assert any("dropped constraint claims ['made_up']" in note for note in result.guard_notes)


def test_the_prompt_names_failing_constraints_and_executed_attempts():
    client = ScriptedClient(_reply())
    attempt = _attempt("ExtraTrees")
    report = _report(_check("max_latency_ms_per_1k", False), _check("min_primary_metric", True))
    reflect(report, client=client, attempts=[attempt], iteration=3, budget_left_s=120.0)
    prompt = client.prompts[0]
    assert "max_latency_ms_per_1k: max_latency_ms_per_1k explained" in prompt
    assert attempt.signature in prompt
    assert "iteration=3" in prompt and "budget_left_s=120.0" in prompt


# --- suggestions ----------------------------------------------------------------


def test_a_suggestion_changes_only_its_own_block():
    base = _plan("LinearModel")
    changed = suggestion_plan(base, ReflectionSuggestion(category="scaler", component="RobustScaler"))
    assert [s.component for s in changed.preprocessing if s.category == "scaler"] == ["RobustScaler"]
    assert [s.component for s in changed.preprocessing if s.category == "encoder"] == ["OneHotEncoder"]
    assert changed.model_family == "LinearModel"


def test_a_suggestion_for_an_executed_configuration_is_not_usable():
    tried = {plan_signature(_plan("ExtraTrees"), categorical_cardinality=5)}
    usable, _ = usable_suggestions(
        [ReflectionSuggestion(category="model_family", component="ExtraTrees")],
        _plan("LinearModel"),
        _profile(),
        iteration=2,
        tried=tried,
    )
    assert usable == []


def test_a_new_configuration_is_usable_and_substitutions_are_reported():
    usable, substitutions = usable_suggestions(
        [
            ReflectionSuggestion(category="model_family", component="ExtraTrees"),
            ReflectionSuggestion(category="model_family", component="MysteryNet"),
        ],
        _plan("LinearModel"),
        _profile(),
        iteration=2,
        tried=set(),
    )
    families = [plan.model_family for _, plan in usable]
    assert "ExtraTrees" in families
    assert all(s.source == "reflection" for s in substitutions)


def test_reflection_suggestions_for_a_glass_box_run_are_resolved_to_interpretable_families():
    profile = PlanProfile(problem_type="classification", n_rows=1_000, categorical_cardinality=5, glass_box=True)
    result = resolve(_plan("GradientBoosting"), profile, source="reflection")
    assert result.plan.model_family in ("LinearModel", "DecisionTree")
