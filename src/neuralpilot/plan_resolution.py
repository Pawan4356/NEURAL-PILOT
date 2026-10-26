"""Plan resolution (Phase 4, Deliverable 8.2).

Turns a planner's `PlanSpec` into one whose every component name is a real
registry entry that can run on this dataset. For each component:

1. an exact name or registered alias is used as is;
2. otherwise, or when the component cannot run on this task, data size or
   constraint, a ranked alternative is substituted, and the substitution is
   recorded as an `UnsupportedRecommendation`;
3. a substitution that changes the task itself is refused: if no safe
   alternative exists, the whole plan is rejected with a message the planner
   can act on.

Pure and deterministic: no LLM, no data access. The only input from the
dataset is a `PlanProfile` the caller builds from the run state.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .registry import (
    REGISTRY,
    Category,
    ComponentSpec,
    components_supporting,
    default_component,
    find_component,
)
from .schemas import CandidateSource, PlanSpec, PreprocessingStep, ProblemType, UnsupportedRecommendation

# Model groups whose trees can use ordinal or target-encoded features.
# Linear models cannot, so an encoder swap across groups is refused for them.
TREE_GROUPS = frozenset({"tree", "bagging_tree", "gbdt"})

# Ranking weights from Deliverable 8.2: task fit, data compatibility, prior
# evidence from memory, and (inverted) resource cost.
_W_TASK, _W_DATA, _W_PRIOR, _W_RESOURCE = 0.4, 0.25, 0.2, 0.15
_RESOURCE_COST = {"light": 0.0, "medium": 0.5, "heavy": 1.0}
_NEUTRAL_PRIOR = 0.5


@dataclass(frozen=True)
class PlanProfile:
    """The facts about the dataset and task that resolution depends on."""

    problem_type: ProblemType
    n_rows: int
    categorical_cardinality: int | None = None
    glass_box: bool = False


@dataclass
class ResolvedPlan:
    plan: PlanSpec
    substitutions: list[UnsupportedRecommendation]


@dataclass
class PlanRejection:
    reason: str
    substitutions: list[UnsupportedRecommendation]


def resolve(
    plan: PlanSpec,
    profile: PlanProfile,
    *,
    iteration: int = 1,
    source: CandidateSource = "planner",
    prior_evidence: dict[str, float] | None = None,
) -> ResolvedPlan | PlanRejection:
    """Map every component of `plan` to a usable registry entry.

    `prior_evidence` maps component names to a score from similar past runs
    (higher is better); components without evidence get a neutral prior."""
    prior = prior_evidence or {}
    substitutions: list[UnsupportedRecommendation] = []

    model = _resolve_model(plan.model_family, profile, iteration, source, prior, substitutions)
    if isinstance(model, PlanRejection):
        return model
    model_group = model.semantic_group

    steps: list[PreprocessingStep] = []
    for step in plan.preprocessing:
        chosen = _resolve_component(
            step.category,
            step.component,
            profile,
            iteration=iteration,
            source=source,
            model_group=model_group,
            prior=prior,
            substitutions=substitutions,
        )
        if isinstance(chosen, PlanRejection):
            return chosen
        if chosen.name != step.component:
            # Parameters were written for the requested component, not the substitute.
            steps.append(step.model_copy(update={"component": chosen.name, "params": {}}))
        else:
            steps.append(step)

    tuner = _resolve_component(
        "tuner", plan.tuning.component, profile, iteration=iteration, source=source, prior=prior,
        substitutions=substitutions, model_group=model_group,
    )
    if isinstance(tuner, PlanRejection):
        return tuner

    resolved = plan.model_copy(
        update={
            "model_family": model.name,
            "preprocessing": steps,
            "tuning": plan.tuning.model_copy(update={"component": tuner.name}),
        }
    )
    return ResolvedPlan(plan=resolved, substitutions=substitutions)


def _resolve_model(
    requested: str,
    profile: PlanProfile,
    iteration: int,
    source: CandidateSource,
    prior: dict[str, float],
    substitutions: list[UnsupportedRecommendation],
) -> ComponentSpec | PlanRejection:
    """Model families get one extra rule: a hard glass-box requirement."""
    chosen = _resolve_component(
        "model_family", requested, profile, iteration=iteration, source=source, prior=prior,
        substitutions=substitutions, model_group=None,
    )
    if isinstance(chosen, PlanRejection):
        return chosen
    if profile.glass_box and chosen.interp_class not in config.GLASS_BOX_CLASSES:
        glass_box = _best_glass_box(profile, prior)
        if glass_box is None:
            return PlanRejection(
                reason="The user requires a glass-box model, but no glass-box model fits this task.",
                substitutions=substitutions,
            )
        _record(
            substitutions, iteration, requested, "model_family", source, "violates_constraint",
            glass_box.name, chosen.semantic_group != glass_box.semantic_group,
        )
        return glass_box
    return chosen


def _best_glass_box(profile: PlanProfile, prior: dict[str, float]) -> ComponentSpec | None:
    options = [
        spec
        for spec in _family_specs(profile.problem_type)
        if spec.interp_class in config.GLASS_BOX_CLASSES
    ]
    if not options:
        return None
    return max(options, key=lambda spec: _score(spec, profile, prior))


def _resolve_component(
    category: Category,
    requested: str,
    profile: PlanProfile,
    *,
    iteration: int,
    source: CandidateSource,
    prior: dict[str, float],
    substitutions: list[UnsupportedRecommendation],
    model_group: str | None,
) -> ComponentSpec | PlanRejection:
    requested_spec = find_component(category, requested)
    if requested_spec is None:
        # Unknown names are not a semantic request we can compare; they go to
        # the category default, exactly as the registry has always done.
        default = default_component(category, categorical_cardinality=profile.categorical_cardinality)
        if _why_unusable(default, profile) is None:
            _record(substitutions, iteration, requested, category, source, "not_in_registry", default.name, False)
            return default
        return _substitute(
            category, default, "not_in_registry", None, profile, iteration, source, prior, substitutions, model_group,
            requested,
        )

    reason = _why_unusable(requested_spec, profile)
    if reason is None:
        return requested_spec
    return _substitute(
        category, requested_spec, reason, requested_spec, profile, iteration, source, prior, substitutions,
        model_group, requested,
    )


def _substitute(
    category: Category,
    spec: ComponentSpec,
    reason: str,
    requested_spec: ComponentSpec | None,
    profile: PlanProfile,
    iteration: int,
    source: CandidateSource,
    prior: dict[str, float],
    substitutions: list[UnsupportedRecommendation],
    model_group: str | None,
    requested: str,
) -> ComponentSpec | PlanRejection:
    alternatives = _ranked_alternatives(category, requested_spec, profile, prior, model_group)
    if not alternatives:
        return PlanRejection(
            reason=(
                f"'{requested}' ({reason.replace('_', ' ')}) has no usable alternative for this "
                f"{profile.problem_type} task. Choose one of: "
                f"{', '.join(components_supporting(category, profile.problem_type))}."
            ),
            substitutions=substitutions,
        )
    chosen = alternatives[0]
    same_group = requested_spec is not None and chosen.semantic_group == requested_spec.semantic_group
    _record(
        substitutions, iteration, requested, category, source, reason, chosen.name,
        requested_spec is not None and not same_group,
    )
    return chosen


def _ranked_alternatives(
    category: Category,
    requested_spec: ComponentSpec | None,
    profile: PlanProfile,
    prior: dict[str, float],
    model_group: str | None,
) -> list[ComponentSpec]:
    """Usable substitutes, best first. Same semantic group always comes first."""
    if requested_spec is None:
        pool = _category_specs(category)
        group = ""
    else:
        names = list(requested_spec.alternatives)
        names += [s.name for s in _category_specs(category) if s.semantic_group == requested_spec.semantic_group]
        pool = [find_component(category, name) for name in dict.fromkeys(names)]
        pool = [spec for spec in pool if spec is not None and spec.name != requested_spec.name]
        group = requested_spec.semantic_group

    usable = [
        spec
        for spec in pool
        if _why_unusable(spec, profile) is None and _may_substitute(category, group, spec, model_group)
    ]
    usable.sort(key=lambda spec: (spec.semantic_group != group, -_score(spec, profile, prior)))
    return usable


def _may_substitute(category: Category, group: str, spec: ComponentSpec, model_group: str | None) -> bool:
    if not group or spec.semantic_group == group:
        return True
    if category == "encoder":
        # One-hot, ordinal and target encodings only mean the same thing to trees.
        return model_group in TREE_GROUPS
    return True


def _why_unusable(spec: ComponentSpec, profile: PlanProfile) -> str | None:
    if profile.problem_type not in spec.supports:
        return "incompatible_task"
    if spec.max_rows_hint is not None and profile.n_rows > spec.max_rows_hint:
        return "over_budget"
    return None


def _score(spec: ComponentSpec, profile: PlanProfile, prior: dict[str, float]) -> float:
    task_fit = 1.0 if profile.problem_type in spec.supports else 0.0
    data_compat = 1.0 if _why_unusable(spec, profile) is None else 0.0
    prior_evidence = prior.get(spec.name, _NEUTRAL_PRIOR)
    resource_cost = _RESOURCE_COST[spec.resource_class]
    return (
        _W_TASK * task_fit
        + _W_DATA * data_compat
        + _W_PRIOR * prior_evidence
        + _W_RESOURCE * (1.0 - resource_cost)
    )


def _family_specs(problem_type: ProblemType) -> list[ComponentSpec]:
    return [spec for spec in REGISTRY["model_family"].values() if problem_type in spec.supports]


def _category_specs(category: Category) -> list[ComponentSpec]:
    return list(REGISTRY[category].values())


def _record(
    substitutions: list[UnsupportedRecommendation],
    iteration: int,
    requested: str,
    category: str,
    source: CandidateSource,
    reason: str,
    alternative: str,
    semantic_change: bool,
) -> None:
    substitutions.append(
        UnsupportedRecommendation(
            iteration=iteration,
            requested=requested,
            category=category,
            source=source,
            reason=reason,
            alternative_used=alternative,
            semantic_change=semantic_change,
        )
    )
