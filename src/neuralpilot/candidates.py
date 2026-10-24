"""Candidate generation (Phase 4, Deliverable 8.3).

Each iteration executes the planner's plan plus a few deterministic
challengers, so the final model is chosen by measurement rather than by
whichever family the planner happened to name. Sources, in priority order:

- baseline: an interpretable model (LinearModel, else DecisionTree);
- profile:  a family suited to the data size (histogram boosting for large
            tables, extra trees for small ones);
- memory:   the plan of a similar past run, if one exists. It is only a
            warm start: it is evaluated like any other candidate.

Challengers never repeat an executed signature, never break a glass-box
requirement (plan resolution substitutes those), and are capped at
`config.MAX_CANDIDATES`. The planner's plan is always kept first.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import config
from .engine import plan_signature
from .plan_resolution import PlanProfile, PlanRejection, resolve
from .schemas import CandidatePlan, CandidateSource, PlanSpec, UnsupportedRecommendation

BASELINE_FAMILIES = ("LinearModel", "DecisionTree")


def build_candidates(
    planner_plan: PlanSpec,
    profile: PlanProfile,
    *,
    iteration: int,
    tried: set[str],
    memory_plans: Iterable[PlanSpec] = (),
    prior_evidence: dict[str, float] | None = None,
    max_candidates: int | None = None,
) -> tuple[list[CandidatePlan], list[UnsupportedRecommendation]]:
    """Return the candidates to execute (planner first) and every component
    substitution made while building them.

    `planner_plan` must already be resolved and must not be in `tried`.
    `max_candidates` defaults to `config.MAX_CANDIDATES`, read at call time."""
    if max_candidates is None:
        max_candidates = config.MAX_CANDIDATES
    cardinality = profile.categorical_cardinality
    candidates = [CandidatePlan(source="planner", plan=planner_plan)]
    seen = {plan_signature(planner_plan, categorical_cardinality=cardinality)} | set(tried)
    substitutions: list[UnsupportedRecommendation] = []

    for source, proposals in _proposals_by_source(planner_plan, profile, memory_plans):
        if len(candidates) >= max_candidates:
            break
        # The first proposal of each source that is usable and new is taken.
        for proposal in proposals:
            result = resolve(
                proposal, profile, iteration=iteration, source=source, prior_evidence=prior_evidence
            )
            if isinstance(result, PlanRejection):
                substitutions.extend(result.substitutions)
                continue
            substitutions.extend(result.substitutions)
            signature = plan_signature(result.plan, categorical_cardinality=cardinality)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(CandidatePlan(source=source, plan=result.plan))
            break
    return candidates, substitutions


def _proposals_by_source(
    planner_plan: PlanSpec, profile: PlanProfile, memory_plans: Iterable[PlanSpec]
) -> list[tuple[CandidateSource, list[PlanSpec]]]:
    return [
        ("baseline", [_with_family(planner_plan, family) for family in BASELINE_FAMILIES]),
        ("profile", [_with_family(planner_plan, family) for family in _profile_families(profile)]),
        ("memory", [_warm_start(planner_plan, past) for past in memory_plans]),
    ]


def _profile_families(profile: PlanProfile) -> list[str]:
    if profile.n_rows > config.HIST_GB_MIN_ROWS:
        return ["HistGradientBoosting"]
    if profile.n_rows <= config.SMALL_DATA_ROWS:
        return ["ExtraTrees"]
    return []


def _with_family(plan: PlanSpec, family: str) -> PlanSpec:
    return plan.model_copy(
        update={"model_family": family, "rationale": f"challenger: {family} from the candidate generator"}
    )


def _warm_start(planner_plan: PlanSpec, past: PlanSpec) -> PlanSpec:
    """A similar run's model and preprocessing, with this iteration's tuning."""
    return past.model_copy(
        update={
            "tuning": planner_plan.tuning,
            "iteration": planner_plan.iteration,
            "rationale": "warm start from a similar past experiment",
        }
    )
