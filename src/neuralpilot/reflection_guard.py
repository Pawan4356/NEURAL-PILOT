"""Guards on Reflection Agent output (Phase 9, Deliverable 9.5).

The Reflection Agent may name a violated constraint or propose a change, but
neither is trusted until it is checked here, deterministically:

- a claimed violation must be a constraint the validation report shows as
  failing (the LLM cannot invent a failure to justify a change);
- a suggestion becomes a full plan, which must resolve against the registry
  and must not repeat an attempt this run has already executed.

Nothing here writes metrics or constraints. The functions only report which
claims and suggestions survive, so the graph can route on them.
"""

from __future__ import annotations

from collections.abc import Iterable

from .engine import plan_signature
from .plan_resolution import PlanProfile, PlanRejection, resolve
from .schemas import (
    ConstraintResult,
    FeatureEngineeringStep,
    PlanSpec,
    PreprocessingStep,
    ReflectionSuggestion,
    TuningStrategy,
    UnsupportedRecommendation,
)
from .validation.constraints import violated_names


def failing_constraints(results: Iterable[ConstraintResult]) -> list[str]:
    """Names of the constraints the report shows as failing."""
    return violated_names(results)


def split_violated_claims(claimed: Iterable[str], failing: Iterable[str]) -> tuple[list[str], list[str]]:
    """(claims that match a failing constraint, claims that do not)."""
    failing = set(failing)
    valid, invalid = [], []
    for name in claimed:
        (valid if name in failing else invalid).append(name)
    return valid, invalid


def suggestion_plan(base: PlanSpec, suggestion: ReflectionSuggestion) -> PlanSpec:
    """`base` with one suggested change applied. The change is not checked
    here; `usable_suggestions` resolves and de-duplicates the result."""
    rationale = f"reflection suggestion: {suggestion.rationale}".rstrip(": ")
    if suggestion.category == "model_family":
        return base.model_copy(update={"model_family": suggestion.component, "rationale": rationale})
    if suggestion.category == "tuner":
        tuning = TuningStrategy(
            component=suggestion.component, n_trials=base.tuning.n_trials, rationale=rationale
        )
        return base.model_copy(update={"tuning": tuning, "rationale": rationale})
    if suggestion.category in ("encoder", "scaler"):
        step = PreprocessingStep(
            category=suggestion.category,
            component=suggestion.component,
            params=suggestion.params,
            rationale=rationale,
        )
        kept = [s for s in base.preprocessing if s.category != suggestion.category]
        return base.model_copy(update={"preprocessing": [*kept, step], "rationale": rationale})
    # feature_engineering: one named step, replacing any step with the same name.
    step = FeatureEngineeringStep(name=suggestion.component, params=suggestion.params, rationale=rationale)
    kept = [s for s in base.feature_engineering if s.name != suggestion.component]
    return base.model_copy(update={"feature_engineering": [*kept, step], "rationale": rationale})


def usable_suggestions(
    suggestions: Iterable[ReflectionSuggestion],
    base: PlanSpec,
    profile: PlanProfile,
    *,
    iteration: int,
    tried: set[str],
) -> tuple[list[tuple[ReflectionSuggestion, PlanSpec]], list[UnsupportedRecommendation]]:
    """The suggestions that resolve against the registry to a configuration not
    yet executed, with the plan each one resolved to, and every component
    substitution made on the way."""
    usable: list[tuple[ReflectionSuggestion, PlanSpec]] = []
    substitutions: list[UnsupportedRecommendation] = []
    seen = set(tried)
    for suggestion in suggestions:
        result = resolve(suggestion_plan(base, suggestion), profile, iteration=iteration, source="reflection")
        if isinstance(result, PlanRejection):
            substitutions.extend(result.substitutions)
            continue
        substitutions.extend(result.substitutions)
        signature = plan_signature(result.plan, categorical_cardinality=profile.categorical_cardinality)
        if signature in seen:
            continue
        seen.add(signature)
        usable.append((suggestion, result.plan))
    return usable, substitutions
