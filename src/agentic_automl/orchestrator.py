"""Orchestration loop (§6) — wires the four agents + deterministic
execution layer into the linear closed loop:

    plan -> verify -> execute -> validate -> reflect -> decide -> (refine/replan/stop)

Only one interaction point exists beyond the initial upload: a single
clarifying question when target-column confidence is low (§7). Everything
else resolves silently through defaults/fallbacks and is surfaced in the
final report's "assumptions" list.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from . import config
from .agents.decision_agent import decide
from .agents.planning_agent import generate_plan
from .agents.reflection_agent import reflect
from .agents.requirement_agent import apply_clarification_answer, needs_clarification, understand_requirement
from .execution import run_pipeline
from .meta_features import DatasetMetaFeatures, extract_meta_features
from .repository import ExperimentRecord, ExperimentRepository, retrieve_similar_experiments
from .schemas import DecisionOutput, PlanSpec, ReflectionOutput, RequirementSpec, ValidationReport
from .scoring import compute_validation, should_generate_shap
from .verification import verify_plan

RunStatus = Literal["running", "awaiting_clarification", "finalized", "failed"]

MAX_VERIFICATION_RETRIES = 3
MODELS_DIR = config.RUNS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IterationLog:
    iteration: int
    plan_summary: dict
    composite_score: float
    accuracy_norm: float
    latency_norm: float
    interp_norm: float
    weakest_block: str
    decision: str
    decision_rationale: str
    fallbacks: list[dict]
    verification_warnings: list[str]


@dataclass
class RunState:
    run_id: str
    dataset_path: Path
    dataset_name: str
    user_goal: str
    df: pd.DataFrame
    status: RunStatus = "running"
    clarification_question: str | None = None
    requirement: RequirementSpec | None = None
    iteration: int = 0
    previous_plan: PlanSpec | None = None
    last_reflection: ReflectionOutput | None = None
    last_verification_failure: str | None = None
    score_history: list[float] = field(default_factory=list)
    weakest_block_history: list[str] = field(default_factory=list)
    iteration_logs: list[IterationLog] = field(default_factory=list)
    rolling_max_latency: float = 0.0
    model_path: str | None = None
    final_shap_summary: dict | None = None
    final_metrics: dict | None = None
    final_decision: DecisionOutput | None = None
    final_report: dict | None = None
    error: str | None = None


def create_run(dataset_path: Path, dataset_name: str, user_goal: str) -> RunState:
    df = pd.read_csv(dataset_path)
    return RunState(
        run_id=str(uuid.uuid4()),
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        user_goal=user_goal,
        df=df,
    )


def _run_requirement_understanding(state: RunState) -> None:
    meta_features = extract_meta_features(state.df, target_column=None)
    spec = understand_requirement(state.user_goal, meta_features)

    if needs_clarification(spec):
        state.requirement = spec
        state.status = "awaiting_clarification"
        state.clarification_question = (
            f"I'm not fully sure which column you want to predict "
            f"(best guess: '{spec.target_column}', confidence {spec.target_confidence:.0%}). "
            f"Which column should be the prediction target? "
            f"Available columns: {', '.join(state.df.columns)}"
        )
        return

    state.requirement = spec
    state.status = "running"


def apply_clarification(state: RunState, answer: str) -> RunState:
    if state.requirement is None:
        raise ValueError("Cannot apply clarification before requirement understanding has run")
    state.requirement = apply_clarification_answer(state.requirement, answer)
    state.status = "running"
    state.clarification_question = None
    return state


def _run_one_iteration(state: RunState, repo: ExperimentRepository) -> None:
    requirement = state.requirement
    assert requirement is not None

    meta_features = extract_meta_features(state.df, target_column=requirement.target_column)
    similar_experiments = retrieve_similar_experiments(meta_features, k=config.RETRIEVAL_TOP_K, repo=repo)

    # --- plan -> verify, retried deterministically on verification failure ---
    plan = None
    verification_warnings: list[str] = []
    for _ in range(MAX_VERIFICATION_RETRIES):
        plan = generate_plan(
            requirement,
            meta_features,
            similar_experiments,
            iteration=state.iteration + 1,
            previous_plan=state.previous_plan,
            reflection=state.last_reflection,
            verification_failure=state.last_verification_failure,
        )
        verification = verify_plan(plan, requirement, state.df)
        if verification.passed:
            verification_warnings = verification.warnings
            state.last_verification_failure = None
            break
        state.last_verification_failure = "; ".join(verification.failures)
        state.previous_plan = plan
    else:
        raise RuntimeError(
            f"Plan repeatedly failed verification after {MAX_VERIFICATION_RETRIES} attempts: "
            f"{state.last_verification_failure}"
        )

    # --- execute ---
    want_shap = should_generate_shap(requirement.constraint_weights, requirement.weight_confidence)
    outcome = run_pipeline(plan, requirement, state.df, generate_shap=want_shap)
    execution_result = outcome.result
    state.rolling_max_latency = max(state.rolling_max_latency, execution_result.latency_seconds)

    # --- validate ---
    validation: ValidationReport = compute_validation(
        raw_metrics=execution_result.metrics,
        latency_seconds=execution_result.latency_seconds,
        rolling_max_latency=state.rolling_max_latency,
        model_family=plan.model_family,
        problem_type=requirement.problem_type,
        weights=requirement.constraint_weights,
        per_block_diagnostics=execution_result.diagnostics,
    )
    state.score_history.append(validation.composite_score)

    # --- reflect ---
    reflection = reflect(validation)
    state.weakest_block_history.append(reflection.weakest_block)

    # --- decide (deterministic guardrails live inside decide()) ---
    state.iteration += 1
    decision = decide(
        composite_score=validation.composite_score,
        score_history=state.score_history,
        weakest_block=reflection.weakest_block,
        weakest_block_history=state.weakest_block_history,
        iteration_count=state.iteration,
        max_iterations=config.MAX_ITERATIONS,
    )

    all_fallbacks = [*execution_result.fallbacks]
    state.iteration_logs.append(
        IterationLog(
            iteration=state.iteration,
            plan_summary={
                "model_family": plan.model_family,
                "preprocessing": [s.component for s in plan.preprocessing],
                "feature_engineering": [s.name for s in plan.feature_engineering],
                "tuning": plan.tuning.component,
            },
            composite_score=validation.composite_score,
            accuracy_norm=validation.accuracy_norm,
            latency_norm=validation.latency_norm,
            interp_norm=validation.interp_norm,
            weakest_block=reflection.weakest_block,
            decision=decision.decision,
            decision_rationale=decision.rationale,
            fallbacks=all_fallbacks,
            verification_warnings=verification_warnings,
        )
    )

    state.previous_plan = plan
    state.last_reflection = reflection if decision.decision == "refine" else None
    state.final_decision = decision
    state.final_metrics = execution_result.metrics

    if decision.decision in ("accept", "stop"):
        _finalize(state, plan, outcome, decision, repo)
    elif decision.decision == "replan":
        state.last_reflection = None  # replan = fresh start, not a targeted revision


def _finalize(state: RunState, plan: PlanSpec, outcome, decision: DecisionOutput, repo: ExperimentRepository) -> None:
    model_path = MODELS_DIR / f"{state.run_id}.joblib"
    joblib.dump(outcome.pipeline, model_path)
    state.model_path = str(model_path)
    state.final_shap_summary = outcome.result.shap_summary

    assumptions = list(state.requirement.assumptions)
    for log in state.iteration_logs:
        for fb in log.fallbacks:
            assumptions.append(f"iteration {log.iteration}: {fb['category']} '{fb['requested']}' unsupported -> used '{fb['used']}'")
        for w in log.verification_warnings:
            assumptions.append(f"iteration {log.iteration}: {w}")

    state.final_report = {
        "run_id": state.run_id,
        "dataset_name": state.dataset_name,
        "target_column": state.requirement.target_column,
        "problem_type": state.requirement.problem_type,
        "model_family": plan.model_family,
        "model_path": state.model_path,
        "metrics": state.final_metrics,
        "composite_score": state.score_history[-1],
        "constraint_weights": state.requirement.constraint_weights.model_dump(),
        "shap_summary": state.final_shap_summary,
        "assumptions": assumptions,
        "final_decision": decision.decision,
        "final_decision_rationale": decision.rationale,
        "run_history": [
            {
                "iteration": log.iteration,
                "plan_summary": log.plan_summary,
                "composite_score": log.composite_score,
                "weakest_block": log.weakest_block,
                "decision": log.decision,
                "decision_rationale": log.decision_rationale,
            }
            for log in state.iteration_logs
        ],
    }
    state.status = "finalized"

    repo.save_run(
        ExperimentRecord.new(
            dataset_name=state.dataset_name,
            problem_type=state.requirement.problem_type,
            target_column=state.requirement.target_column,
            meta_features=extract_meta_features(state.df, state.requirement.target_column).to_dict(),
            plan=plan.model_dump(),
            scores=state.score_history,
            weakest_block_history=state.weakest_block_history,
            final_decision=decision.decision,
            final_composite_score=state.score_history[-1],
        )
    )


def run_to_completion(state: RunState, repo: ExperimentRepository | None = None) -> RunState:
    """Advance a RunState through the loop until it finalizes or needs the
    one allowed clarifying question (§7). Safe to call again after
    apply_clarification() to resume.
    """
    owns_repo = repo is None
    repo = repo or ExperimentRepository()
    try:
        if state.requirement is None:
            _run_requirement_understanding(state)
            if state.status == "awaiting_clarification":
                return state

        while state.status == "running":
            _run_one_iteration(state, repo)
        return state
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        return state
    finally:
        if owns_repo:
            repo.close()
