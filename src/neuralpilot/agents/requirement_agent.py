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

import pandas as pd

from .. import config
from ..llm_client import LLMClient, get_default_client
from ..observability import traced
from ..meta_features import DatasetMetaFeatures
from ..profiling import derive_problem_type
from ..schemas import ConstraintWeights, RequirementSpec


class InvalidClarification(ValueError):
    """The clarification answer does not name a column of the dataset."""

    def __init__(self, answer: str, columns: list[str]):
        self.answer = answer
        self.columns = columns
        super().__init__(
            f"'{answer}' is not a column in the dataset. Available columns: {', '.join(columns)}"
        )

SYSTEM_PROMPT = """You are the Requirement Understanding Agent in the Neural Pilot system.
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
  "assumptions": [string, ...],
  "primary_metric": null | "accuracy" | "balanced_accuracy" | "f1_weighted" | "f1_macro" | "roc_auc" | "r2" | "rmse" | "mae",
  "time_column": null | string,
  "group_column": null | string
}
- primary_metric: set only if the user names a metric; otherwise null.
- time_column / group_column: set only if the user explicitly asks for a time-ordered or \
grouped evaluation (e.g. "predict future months", "customers must not leak across splits"); \
otherwise null.
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


@traced("requirement_agent", tags=["agent"])
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


def match_column(answer: str, columns: list[str]) -> str | None:
    """Exact match first, then case/whitespace-insensitive."""
    answer = answer.strip()
    if answer in columns:
        return answer
    folded = answer.casefold()
    matches = [c for c in columns if str(c).strip().casefold() == folded]
    return matches[0] if len(matches) == 1 else None


def apply_clarification_answer(
    spec: RequirementSpec, target_column_answer: str, df: pd.DataFrame | None = None
) -> RequirementSpec:
    """§7: the single allowed mid-run user interaction. The clarifying
    question asks the user to name the target column directly, so applying
    the answer is deterministic — no second LLM call needed.

    With `df`, the answer must match a column (case-insensitively) or
    InvalidClarification is raised, and problem_type is re-derived from the
    chosen column instead of keeping the guess made for a different column.
    """
    target = target_column_answer.strip()
    assumptions = [*spec.assumptions, "target_column confirmed by user via clarifying question"]
    update: dict = {"target_column": target, "target_confidence": 1.0}
    if df is not None:
        columns = [str(c) for c in df.columns]
        matched = match_column(target, columns)
        if matched is None:
            raise InvalidClarification(target, columns)
        problem_type = derive_problem_type(df[matched])
        if problem_type != spec.problem_type:
            assumptions.append(
                f"problem_type re-derived as '{problem_type}' from target column '{matched}'"
            )
        update = {"target_column": matched, "target_confidence": 1.0, "problem_type": problem_type}
    updated = spec.model_copy(update=update)
    updated.assumptions = assumptions
    return updated
