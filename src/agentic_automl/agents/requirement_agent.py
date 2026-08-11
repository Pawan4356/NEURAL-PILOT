"""Requirement Understanding Agent (§3.1).

Input: raw user prompt + dataset meta-features (column names, dtypes,
row/col count — never raw rows). Output: target column, problem type,
constraint weights, and confidences.

Clarification rule: if target_column confidence < 0.7, the orchestrator
must ask the user one clarifying question before proceeding (§7). If
weight_confidence < 0.7, we do NOT ask — we silently fall back to default
weights and record the assumption in the final report.
"""

from __future__ import annotations

import json

from .. import config
from ..llm_client import LLMClient, get_default_client
from ..meta_features import DatasetMetaFeatures
from ..schemas import ConstraintWeights, RequirementSpec

SYSTEM_PROMPT = """You are the Requirement Understanding Agent in an agentic AutoML system.
You never write or execute code. You only read the user's plain-English goal and the \
dataset's meta-features (column names, dtypes, cardinality — never raw rows) and produce \
a structured specification.

Rules:
- Infer the target_column from the user's goal and the column names/dtypes. If uncertain, \
still pick your best guess and reflect your uncertainty honestly in target_confidence (0-1).
- Infer problem_type ("classification" or "regression") from the target column's dtype and \
cardinality: low-cardinality/non-numeric targets are usually classification; \
high-cardinality numeric targets are usually regression.
- Infer constraint_weights (accuracy/latency/interpretability, each 0-1) from explicit \
priority language in the user's goal (e.g. "fast", "explainable", "must be interpretable", \
"as accurate as possible"). If no explicit priority language is present, use the default \
0.7/0.15/0.15 split and set weight_confidence low (< 0.7).
- Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{
  "target_column": string,
  "target_confidence": number (0-1),
  "problem_type": "classification" | "regression",
  "constraint_weights": {"accuracy": number, "latency": number, "interpretability": number},
  "weight_confidence": number (0-1),
  "assumptions": [string, ...]
}
"""


def _user_prompt(user_goal: str, meta_features: DatasetMetaFeatures) -> str:
    columns_summary = [
        {"name": c["name"], "dtype": c["dtype"], "n_unique": c["n_unique"], "missing_pct": round(c["missing_pct"], 2)}
        for c in meta_features.columns
    ]
    return (
        f"User goal: {user_goal}\n\n"
        f"Dataset shape: {meta_features.n_rows} rows x {meta_features.n_cols} columns\n"
        f"Columns: {json.dumps(columns_summary)}\n"
    )


def understand_requirement(
    user_goal: str, meta_features: DatasetMetaFeatures, client: LLMClient | None = None
) -> RequirementSpec:
    client = client or get_default_client()
    spec = client.complete_json(
        SYSTEM_PROMPT, _user_prompt(user_goal, meta_features), RequirementSpec
    )

    if spec.weight_confidence < config.WEIGHT_CONFIDENCE_THRESHOLD:
        spec.constraint_weights = ConstraintWeights(**config.DEFAULT_CONSTRAINT_WEIGHTS)
        spec.assumptions.append(
            "weight_confidence below threshold — fell back to default constraint weights "
            f"({config.DEFAULT_CONSTRAINT_WEIGHTS['accuracy']}/"
            f"{config.DEFAULT_CONSTRAINT_WEIGHTS['latency']}/"
            f"{config.DEFAULT_CONSTRAINT_WEIGHTS['interpretability']})"
        )

    return spec


def needs_clarification(spec: RequirementSpec) -> bool:
    return spec.target_confidence < config.TARGET_CONFIDENCE_THRESHOLD


def apply_clarification_answer(spec: RequirementSpec, target_column_answer: str) -> RequirementSpec:
    """§7: the single allowed mid-run user interaction. The clarifying
    question asks the user to name the target column directly, so applying
    the answer is a deterministic overwrite — no second LLM call needed.
    """
    updated = spec.model_copy(update={"target_column": target_column_answer.strip(), "target_confidence": 1.0})
    updated.assumptions = [*spec.assumptions, "target_column confirmed by user via clarifying question"]
    return updated
