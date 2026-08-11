"""Planning Agent (§3.2).

Produces one structured pipeline plan: preprocessing steps, feature
engineering steps, a candidate model family, and a tuning strategy, each
with a one-line rationale. On refine (loop iteration > 1), it receives the
Reflection Agent's target block and revises only that block, keeping the
rest of the plan fixed (§3.2, §6 step 9).
"""

from __future__ import annotations

import json

from ..llm_client import LLMClient, get_default_client
from ..meta_features import DatasetMetaFeatures
from ..registry import list_components
from ..repository import ExperimentRecord
from ..schemas import PlanSpec, ReflectionOutput, RequirementSpec

SYSTEM_PROMPT = f"""You are the Planning Agent in an agentic AutoML system.
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

Produce exactly one pipeline plan (no A/B/C alternatives). Give each step a one-line \
rationale. Respond with ONLY a JSON object matching this schema, no prose, no markdown \
fences:
{{
  "preprocessing": [{{"category": "encoder"|"scaler", "component": string, "params": object, "rationale": string}}, ...],
  "feature_engineering": [{{"name": string, "params": object, "rationale": string}}, ...],
  "model_family": string,
  "tuning": {{"component": string, "n_trials": integer, "rationale": string}},
  "rationale": string
}}
Always include exactly one "encoder" and one "scaler" preprocessing step.
"""


def _format_similar_experiments(similar_experiments: list[ExperimentRecord]) -> str:
    if not similar_experiments:
        return "None available yet."
    lines = []
    for rec in similar_experiments:
        lines.append(
            f"- dataset={rec.dataset_name}, model_family={rec.plan.get('model_family')}, "
            f"final_score={rec.final_composite_score:.3f}, decision={rec.final_decision}"
        )
    return "\n".join(lines)


def _base_user_prompt(
    requirement: RequirementSpec, meta_features: DatasetMetaFeatures, similar_experiments: list[ExperimentRecord]
) -> str:
    return (
        f"Requirement spec: {requirement.model_dump_json()}\n\n"
        f"Dataset meta-features: rows={meta_features.n_rows}, cols={meta_features.n_cols}, "
        f"n_classes={meta_features.n_classes}, missing_pct={meta_features.missing_pct:.2f}, "
        f"categorical_ratio={meta_features.categorical_ratio:.2f}\n\n"
        f"Top-k similar past experiments (for reference, not binding):\n"
        f"{_format_similar_experiments(similar_experiments)}\n"
    )


def generate_plan(
    requirement: RequirementSpec,
    meta_features: DatasetMetaFeatures,
    similar_experiments: list[ExperimentRecord],
    *,
    client: LLMClient | None = None,
    iteration: int = 1,
    previous_plan: PlanSpec | None = None,
    reflection: ReflectionOutput | None = None,
    verification_failure: str | None = None,
) -> PlanSpec:
    client = client or get_default_client()
    user_prompt = _base_user_prompt(requirement, meta_features, similar_experiments)

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

    plan = client.complete_json(SYSTEM_PROMPT, user_prompt, PlanSpec)
    plan.iteration = iteration
    return plan
