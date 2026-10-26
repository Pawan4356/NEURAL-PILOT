"""Planning Agent (§3.2).

Produces one structured pipeline plan: preprocessing steps, feature
engineering steps, a candidate model family, and a tuning strategy, each
with a one-line rationale. On refine (loop iteration > 1), it receives the
Reflection Agent's target block and revises only that block, keeping the
rest of the plan fixed (§3.2, §6 step 9).

Every prompt lists the plans already executed in this run (with their CV
scores) so the agent does not repeat them; on replan it is told to change
strategy. The orchestrator also rejects exact repeats by signature.
"""

from __future__ import annotations

import json

from .. import config
from ..llm_client import LLMClient, get_default_client
from ..observability import traced
from ..meta_features import DatasetMetaFeatures
from ..registry import list_components
from ..repository import ExperimentRecord, SimilarExperiment
from ..schemas import AttemptSummary, PlanSpec, ReflectionOutput, RequirementSpec, UnsupportedRecommendation

SYSTEM_PROMPT = f"""You are the Planning Agent in the Neural Pilot system.
You never write or execute code. You only select and parametrize trusted, pre-built \
components by name from a fixed registry. The Execution Engine will run whatever you \
choose deterministically.

Available components by category (choosing a name outside this list is safe — the \
Execution Engine will deterministically fall back to a sensible default and log it — but \
prefer a listed name whenever it fits):
- encoder: {list_components("encoder")}
- scaler: {list_components("scaler")}
- model_family: {list_components("model_family")}
- tuner: {list_components("tuner")}
- feature_engineering (optional, name field): PolynomialFeatures, SelectKBest, PCA

Supported params (anything else is ignored): OneHotEncoder {{min_frequency, max_categories}}; \
StandardScaler {{with_mean, with_std}}; MinMaxScaler {{feature_range, clip}}; RobustScaler \
{{with_centering, with_scaling, quantile_range}}; PolynomialFeatures {{degree <= 2, \
interaction_only}} (only applied when there are <= {config.POLYNOMIAL_MAX_INPUT_WIDTH} encoded \
features); SelectKBest {{k}}; PCA {{n_components}}. n_trials is capped at {config.HPO_MAX_TRIALS}.

Scores you see are cross-validated on the training split for this dataset. Measured results \
from the current run always outweigh similar past experiments and the documentation passages, \
which describe how components behave in general.

Produce exactly one pipeline plan (no A/B/C alternatives). Give each step a one-line \
rationale. Respond with ONLY a JSON object matching this schema, no prose, no markdown \
fences:
{{
  "preprocessing": [{{"category": "encoder"|"scaler", "component": string, "params": object, "rationale": string}}, ...],
  "feature_engineering": [{{"name": string, "params": object, "rationale": string}}, ...],
  "model_family": string,
  "tuning": {{"component": string, "n_trials": integer, "rationale": string}},
  "rationale": string,
  "evidence_ids": [string]
}}
"evidence_ids" lists the ids of the documentation passages (if any are given) that your plan relies on; \
use only ids that are listed.
Always include exactly one "encoder" and one "scaler" preprocessing step.
"""


def _format_similar_experiments(
    similar_experiments: list[SimilarExperiment] | list[ExperimentRecord],
) -> str:
    if not similar_experiments:
        return "None similar enough to be useful."
    lines = []
    for item in similar_experiments:
        rec, similarity = item if isinstance(item, tuple) else (item, None)
        score = f"{rec.final_composite_score:.3f}" if rec.final_composite_score is not None else "n/a"
        sim = f"similarity={similarity:.2f}, " if similarity is not None else ""
        lines.append(
            f"- {sim}dataset={rec.dataset_name}, model_family={rec.plan.get('model_family')}, "
            f"final_score={score}, decision={rec.final_decision}"
        )
    return "\n".join(lines)


def _format_attempts(attempts: list[AttemptSummary]) -> str:
    lines = []
    for a in attempts:
        lines.append(
            f"- signature={a.signature} (iteration {a.iteration}): model_family={a.model_family}, "
            f"preprocessing={a.preprocessing}, feature_engineering={a.feature_engineering or []}, "
            f"tuning={a.tuning}, CV {a.primary_metric}={a.cv_mean:.4f} ± {a.cv_std:.4f}, "
            f"composite={a.composite_score:.4f}"
        )
    return "\n".join(lines)


def _format_unsupported(items: list[UnsupportedRecommendation]) -> str:
    """Components this run already asked for that the registry could not use."""
    lines = [
        f"- {u.category} '{u.requested}' is not usable ({u.reason.replace('_', ' ')}); "
        f"'{u.alternative_used}' was used instead"
        for u in items
    ]
    return (
        "Components the registry could not use in this run (prefer the listed alternative, "
        "or a name from the available list):\n" + "\n".join(lines)
    )


def _format_passages(passages: list[dict]) -> str:
    lines = [f"- id={p['id']} ({p['title']}, section: {p['section']}): {p['text']}" for p in passages]
    return "Documentation passages (curated and official):\n" + "\n".join(lines)


def _base_user_prompt(
    requirement: RequirementSpec,
    meta_features: DatasetMetaFeatures,
    similar_experiments: list[SimilarExperiment] | list[ExperimentRecord],
    dataset_profile: dict | None = None,
) -> str:
    prompt = (
        f"Requirement spec: {requirement.model_dump_json()}\n\n"
        f"Dataset meta-features: rows={meta_features.n_rows}, cols={meta_features.n_cols}, "
        f"n_classes={meta_features.n_classes}, missing_pct={meta_features.missing_pct:.2f}, "
        f"categorical_ratio={meta_features.categorical_ratio:.2f}, "
        f"max_categorical_cardinality={meta_features.max_cardinality}"
    )
    if meta_features.minority_class_ratio is not None and requirement.problem_type == "classification":
        prompt += f", minority_class_share={meta_features.minority_class_ratio:.3f}"
    prompt += "\n"
    if dataset_profile:
        prompt += f"Dataset profile (deterministic): {json.dumps(dataset_profile, default=str)}\n"
    prompt += (
        f"\nTop-k similar past experiments (for reference, not binding):\n"
        f"{_format_similar_experiments(similar_experiments)}\n"
    )
    return prompt


@traced(
    "planning_agent",
    tags=["agent"],
    metadata_fn=lambda *a, **kw: {"iteration": kw.get("iteration", 1), "replan": kw.get("replan", False)},
)
def generate_plan(
    requirement: RequirementSpec,
    meta_features: DatasetMetaFeatures,
    similar_experiments: list[SimilarExperiment] | list[ExperimentRecord],
    *,
    client: LLMClient | None = None,
    iteration: int = 1,
    previous_plan: PlanSpec | None = None,
    reflection: ReflectionOutput | None = None,
    verification_failure: str | None = None,
    attempts: list[AttemptSummary] | None = None,
    replan: bool = False,
    rejection: str | None = None,
    dataset_profile: dict | None = None,
    unsupported: list[UnsupportedRecommendation] | None = None,
    passages: list[dict] | None = None,
) -> PlanSpec:
    client = client or get_default_client()
    user_prompt = _base_user_prompt(requirement, meta_features, similar_experiments, dataset_profile)
    if passages:
        user_prompt += f"\n{_format_passages(passages)}\n"

    if unsupported:
        user_prompt += f"\n{_format_unsupported(unsupported)}\n"
    if attempts:
        user_prompt += (
            f"\nAlready tried in this run (do not repeat any of these exact configurations):\n"
            f"{_format_attempts(attempts)}\n"
        )
    if rejection is not None:
        user_prompt += f"\nYour previous answer was rejected: {rejection}\n"
    if replan:
        user_prompt += (
            f"\nThis is a REPLAN iteration ({iteration}). The current strategy is not working: "
            f"propose a materially different plan — a different model_family and/or "
            f"preprocessing from those already tried.\n"
        )

    if verification_failure is not None and previous_plan is not None:
        user_prompt += (
            f"\nThe previous plan failed deterministic verification with this reason:\n"
            f"{verification_failure}\n"
            f"Previous plan: {previous_plan.model_dump_json()}\n"
            f"Produce a corrected plan that fixes this specific failure. Keep everything else "
            f"as close to the previous plan as reasonable.\n"
        )
    elif reflection is not None and previous_plan is not None:
        user_prompt += (
            f"\nThis is a refine iteration ({iteration}). The Reflection Agent identified "
            f"'{reflection.weakest_block}' as the weakest block, with rationale: "
            f"{reflection.rationale}\n"
            f"Previous plan: {previous_plan.model_dump_json()}\n"
            f"Revise ONLY the '{reflection.weakest_block}' block. Keep every other block "
            f"(preprocessing/feature_engineering/model_family/tuning not named as weakest) "
            f"identical to the previous plan.\n"
        )
        if reflection.suggestions:
            suggested = "; ".join(
                f"{s.category} {s.component} {s.params} ({s.rationale})" for s in reflection.suggestions
            )
            user_prompt += (
                f"The Reflection Agent suggested these changes (already checked against the registry and "
                f"not yet executed): {suggested}\n"
            )

    plan = client.complete_json(SYSTEM_PROMPT, user_prompt, PlanSpec)
    plan.iteration = iteration
    return plan
