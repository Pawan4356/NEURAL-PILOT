"""Deterministic selection rules used by the orchestration graph.

- `is_better`: which of two attempts is kept as the best so far (audit G1).
- `next_untried_plan`: the fallback used when the planner keeps repeating
  a configuration that was already executed (audit G3).

Both are pure functions of measured data, so they are easy to test and
never depend on an LLM.
"""

from __future__ import annotations

import math

from . import config
from .engine import plan_signature
from .engine.evaluation import greater_is_better
from .registry import list_components
from .schemas import AttemptSummary, PlanSpec, PreprocessingStep, RequirementSpec

# Order in which the deterministic fallback tries untried model families.
FALLBACK_MODEL_ORDER = ["RandomForest", "GradientBoosting", "LinearModel", "XGBoost", "DecisionTree", "CatBoost"]
INTERPRETABLE_MODEL_ORDER = ["LinearModel", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "CatBoost"]


def is_better(new: AttemptSummary, best: AttemptSummary | None, n_folds: int) -> bool:
    """A feasible attempt beats an infeasible one whatever its metric (Phase 5).
    Among equally feasible attempts, the primary-metric CV mean decides;
    attempts within one standard error of each other are separated by the
    composite score (user priorities)."""
    if best is None:
        return True
    if new.feasible != best.feasible:
        return new.feasible
    sign = 1.0 if greater_is_better(new.primary_metric) else -1.0
    diff = sign * (new.cv_mean - best.cv_mean)
    standard_error = max(new.cv_std, best.cv_std) / math.sqrt(max(n_folds, 1))
    if abs(diff) <= standard_error:
        return new.composite_score > best.composite_score
    return diff > 0


def next_untried_plan(
    plan: PlanSpec, requirement: RequirementSpec, tried: set[str], categorical_cardinality: int | None
) -> PlanSpec | None:
    """The closest variant of `plan` whose signature has not been executed —
    first another model family, then other encoder/scaler combinations.
    None when every registry variant has been tried."""
    prefers_interpretability = requirement.constraint_weights.normalized().interpretability > (
        config.DEFAULT_CONSTRAINT_WEIGHTS["interpretability"] + 1e-9
    )
    order = INTERPRETABLE_MODEL_ORDER if prefers_interpretability else FALLBACK_MODEL_ORDER

    def untried(candidate: PlanSpec) -> bool:
        return plan_signature(candidate, categorical_cardinality=categorical_cardinality) not in tried

    for model in order:
        candidate = plan.model_copy(
            update={"model_family": model, "rationale": f"deterministic fallback: next untried model family ({model})"}
        )
        if untried(candidate):
            return candidate
    for model in order:
        for encoder in list_components("encoder"):
            for scaler in list_components("scaler"):
                candidate = plan.model_copy(
                    update={
                        "model_family": model,
                        "preprocessing": [
                            PreprocessingStep(category="encoder", component=encoder),
                            PreprocessingStep(category="scaler", component=scaler),
                        ],
                        "rationale": f"deterministic fallback: {model} with {encoder} + {scaler}",
                    }
                )
                if untried(candidate):
                    return candidate
    return None


def plan_summary(plan: PlanSpec) -> dict:
    return {
        "model_family": plan.model_family,
        "preprocessing": [s.component for s in plan.preprocessing],
        "feature_engineering": [s.name for s in plan.feature_engineering],
        "tuning": plan.tuning.component,
    }
