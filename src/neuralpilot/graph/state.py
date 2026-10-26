"""Graph state for one NeuralPilot run.

Everything in `NPState` is a primitive, a dict or a Pydantic model, so the
SQLite checkpointer can save it after every node. DataFrames, prepared
splits and fitted pipelines never go into state; nodes fetch them from
`graph.resources` by dataset fingerprint or run id.

Fields marked "appended" use a reducer: a node returns only the new items
and LangGraph adds them to the list.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from ..schemas import (
    AttemptSummary,
    BestAttempt,
    CandidatePlan,
    DecisionOutput,
    ExecutionResult,
    IterationLog,
    PlanSpec,
    ReflectionOutput,
    RequirementSpec,
    UnsupportedRecommendation,
    ValidationReport,
)

RunStatus = Literal["running", "awaiting_clarification", "finalized", "failed"]


class NPState(TypedDict, total=False):
    # --- identity and inputs (set once when the run is created) ---
    run_id: str
    dataset_path: str
    dataset_name: str
    dataset_fingerprint: str
    load_report: dict
    user_goal: str
    seed: int
    budget_s: float  # wall-clock budget, counted from data preparation
    max_iterations: int
    status: RunStatus

    # --- requirement understanding and clarification ---
    requirement: RequirementSpec | None
    clarification_question: str | None
    clarification_rounds: int

    # --- prepared data (summaries only) ---
    data_summary: dict
    data_notes: list[str]
    categorical_cardinality: int | None
    meta_features: dict  # DatasetMetaFeatures as a dict, target-aware
    budget_started_at: float  # epoch seconds

    # --- knowledge ---
    similar_experiments: list[dict]  # [{"run_id", "similarity"}]
    knowledge_passages: list[dict]  # documentation passages the planner may cite (Phase 8)

    # --- planning (reset at the start of every iteration) ---
    plan_attempt: int
    duplicate_rejections: int
    plan_rejection: str | None
    failed_plan: PlanSpec | None
    verification_failure: str | None
    candidate_plan: PlanSpec | None  # planner output, not yet checked
    plan: PlanSpec | None  # verified, resolved and not a repeat: ready to execute
    plan_source: str  # "planner" or "deterministic_fallback"
    verification_warnings: list[str]
    candidates: list[CandidatePlan] | None  # plan + challengers, executed this iteration
    unsupported_recommendations: Annotated[list[UnsupportedRecommendation], operator.add]  # appended

    # --- carried from one iteration to the next ---
    previous_plan: PlanSpec | None
    last_reflection: ReflectionOutput | None  # only kept after a "refine"
    replan_requested: bool

    # --- execution and validation ---
    attempts: Annotated[list[AttemptSummary], operator.add]  # appended
    score_history: Annotated[list[float], operator.add]  # appended
    weakest_block_history: Annotated[list[str], operator.add]  # appended
    iteration_logs: Annotated[list[IterationLog], operator.add]  # appended
    last_execution: ExecutionResult | None  # the planner's plan this iteration
    last_validation: ValidationReport | None  # the planner's plan this iteration
    last_summary: AttemptSummary | None  # the planner's plan this iteration
    iteration_candidates: list[dict]  # per-candidate outcomes this iteration (for the log)
    reflection: ReflectionOutput | None  # this iteration's reflection
    best: BestAttempt | None

    # --- control ---
    iteration: int  # completed iterations
    decision: DecisionOutput | None

    # --- output ---
    mlflow_run_id: str | None  # parent run in the MLflow tracking store
    holdout_evaluations: int
    model_path: str | None
    final_metrics: dict | None
    final_report: dict | None
    error: str | None
    failed_node: str | None


# Types the checkpointer may rebuild from msgpack. Listing them keeps
# deserialisation to known classes instead of "anything importable".
CHECKPOINT_TYPES = (
    AttemptSummary,
    BestAttempt,
    CandidatePlan,
    DecisionOutput,
    ExecutionResult,
    IterationLog,
    PlanSpec,
    ReflectionOutput,
    RequirementSpec,
    UnsupportedRecommendation,
    ValidationReport,
)


def plan_reset() -> dict:
    """Per-iteration planning fields, cleared before each new plan."""
    return {
        "plan_attempt": 0,
        "duplicate_rejections": 0,
        "plan_rejection": None,
        "failed_plan": None,
        "verification_failure": None,
        "candidate_plan": None,
        "plan": None,
        "plan_source": "planner",
        "verification_warnings": [],
        "candidates": None,
        "iteration_candidates": [],
    }
