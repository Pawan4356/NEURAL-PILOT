"""Orchestration loop (§6) — wires the four agents + deterministic
execution layer into the linear closed loop:

    plan -> verify -> execute -> validate -> reflect -> decide -> (refine/replan/stop)

Only one interaction point exists beyond the initial upload: a single
clarifying question when target-column confidence is low (§7). Everything
else resolves silently through defaults/fallbacks and is surfaced in the
final report's "assumptions" list.

Evaluation protocol (audit G1, G7): every attempt is measured by
cross-validation on the training split; the best attempt so far is kept;
at the end that attempt — not merely the last one — is scored once on the
holdout, saved and reported. Plans already executed in the run are
rejected by signature (audit G3), and the run stops at its wall-clock
budget (audit G9).
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Literal

import joblib
import pandas as pd

from . import config, provenance
from .agents.decision_agent import decide
from .agents.planning_agent import generate_plan
from .agents.reflection_agent import reflect
from .agents.requirement_agent import (
    apply_clarification_answer,
    match_column,
    needs_clarification,
    understand_requirement,
)
from .engine import AttemptBudget, AttemptOutcome, PreparedData, load_dataset, plan_signature, prepare_dataset, run_attempt
from .engine.evaluation import greater_is_better
from .engine.explainability import shap_summary
from .engine.runner import holdout_evaluate
from .meta_features import extract_meta_features
from .registry import list_components
from .repository import ExperimentRecord, ExperimentRepository, retrieve_similar_experiments
from .schemas import (
    AttemptSummary,
    DecisionOutput,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    RequirementSpec,
    ValidationReport,
)
from .scoring import compute_validation, should_generate_shap
from .verification import verify_plan

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "awaiting_clarification", "finalized", "failed"]

MAX_VERIFICATION_RETRIES = 3
MODELS_DIR = config.RUNS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Order in which the deterministic fallback tries untried model families.
_FALLBACK_MODEL_ORDER = ["RandomForest", "GradientBoosting", "LinearModel", "XGBoost", "DecisionTree", "CatBoost"]
_INTERPRETABLE_MODEL_ORDER = ["LinearModel", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "CatBoost"]


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
    signature: str = ""
    plan_source: str = "planner"  # "planner" or "deterministic_fallback"
    primary_metric: str = ""
    cv_mean: float = 0.0
    cv_std: float = 0.0
    latency_ms_per_1k: float = 0.0
    decision_forced: bool = False
    plan_adjustments: list[str] = field(default_factory=list)


@dataclass
class BestAttempt:
    iteration: int
    plan: PlanSpec
    outcome: AttemptOutcome
    validation: ValidationReport
    summary: AttemptSummary


@dataclass
class RunState:
    run_id: str
    dataset_path: Path
    dataset_name: str
    user_goal: str
    df: pd.DataFrame
    status: RunStatus = "running"
    current_task: str = "Starting run"
    activity_log: list[dict[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    clarification_question: str | None = None
    requirement: RequirementSpec | None = None
    iteration: int = 0
    previous_plan: PlanSpec | None = None
    last_reflection: ReflectionOutput | None = None
    last_verification_failure: str | None = None
    replan_requested: bool = False
    score_history: list[float] = field(default_factory=list)
    weakest_block_history: list[str] = field(default_factory=list)
    iteration_logs: list[IterationLog] = field(default_factory=list)
    attempts: list[AttemptSummary] = field(default_factory=list)
    best: BestAttempt | None = None
    prepared: PreparedData | None = None
    dataset_fingerprint: str | None = None
    load_report: dict = field(default_factory=dict)
    budget_s: float = config.RUN_BUDGET_S
    cancel_event: threading.Event = field(default_factory=threading.Event)
    holdout_evaluations: int = 0
    model_path: str | None = None
    final_shap_summary: dict | None = None
    final_metrics: dict | None = None
    final_decision: DecisionOutput | None = None
    final_report: dict | None = None
    error: str | None = None

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def budget_left_s(self) -> float:
        return self.budget_s - self.elapsed_s()


def _record_activity(state: RunState, task: str, tag: str = "agent") -> None:
    elapsed = time.monotonic() - state.started_at
    state.activity_log.append(
        {
            "time": f"{elapsed // 60:02.0f}:{elapsed % 60:04.1f}",
            "tag": tag,
            "message": task,
        }
    )


def create_run(dataset_path: Path, dataset_name: str, user_goal: str) -> RunState:
    """Load the dataset (raises engine.DatasetLoadError with an actionable
    message on unreadable files) and create a run."""
    loaded = load_dataset(dataset_path)
    return RunState(
        run_id=str(uuid.uuid4()),
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        user_goal=user_goal,
        df=loaded.df,
        dataset_fingerprint=loaded.fingerprint,
        load_report=loaded.report,
    )


def _ask_for_target(state: RunState, spec: RequirementSpec, reason: str) -> None:
    state.requirement = spec
    state.status = "awaiting_clarification"
    state.current_task = "Waiting for target column clarification"
    state.clarification_question = (
        f"{reason} Which column should be the prediction target? "
        f"Available columns: {', '.join(map(str, state.df.columns))}"
    )


def _run_requirement_understanding(state: RunState) -> None:
    state.current_task = "Understanding requirements and inspecting dataset"
    _record_activity(state, state.current_task, "requirement")
    logger.info("run=%s task=%s", state.run_id, state.current_task)
    meta_features = extract_meta_features(state.df, target_column=None)
    spec = understand_requirement(state.user_goal, meta_features)

    matched = match_column(spec.target_column, [str(c) for c in state.df.columns])
    if matched is None:
        _ask_for_target(state, spec, f"I couldn't find a column named '{spec.target_column}'.")
        return
    spec.target_column = matched

    if needs_clarification(spec):
        _ask_for_target(
            state,
            spec,
            f"I'm not fully sure which column you want to predict "
            f"(best guess: '{spec.target_column}', confidence {spec.target_confidence:.0%}).",
        )
        return

    state.requirement = spec
    state.status = "running"


def apply_clarification(state: RunState, answer: str) -> RunState:
    """Raises InvalidClarification (state unchanged) if `answer` is not a column."""
    if state.requirement is None:
        raise ValueError("Cannot apply clarification before requirement understanding has run")
    state.requirement = apply_clarification_answer(state.requirement, answer, state.df)
    state.status = "running"
    state.current_task = "Resuming after target column clarification"
    state.clarification_question = None
    return state


def _ensure_prepared(state: RunState) -> PreparedData:
    if state.prepared is None:
        state.current_task = "Profiling columns and checking for leakage"
        _record_activity(state, state.current_task, "execute")
        state.prepared = prepare_dataset(state.df, state.requirement, seed=config.SEED)
        for column, reason in state.prepared.excluded_columns.items():
            logger.info("run=%s excluded column=%s reason=%s", state.run_id, column, reason)
    return state.prepared


def _plan_summary(plan: PlanSpec) -> dict:
    return {
        "model_family": plan.model_family,
        "preprocessing": [s.component for s in plan.preprocessing],
        "feature_engineering": [s.name for s in plan.feature_engineering],
        "tuning": plan.tuning.component,
    }


def _next_untried_plan(
    plan: PlanSpec, requirement: RequirementSpec, tried: set[str], categorical_cardinality: int | None
) -> PlanSpec | None:
    """Deterministic fallback: the closest variant of `plan` whose signature
    has not been executed — first another model family, then other
    encoder/scaler combinations."""
    prefers_interpretability = requirement.constraint_weights.normalized().interpretability > (
        config.DEFAULT_CONSTRAINT_WEIGHTS["interpretability"] + 1e-9
    )
    order = _INTERPRETABLE_MODEL_ORDER if prefers_interpretability else _FALLBACK_MODEL_ORDER

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


def _obtain_plan(
    state: RunState,
    requirement: RequirementSpec,
    meta_features,
    similar_experiments,
    set_task,
) -> tuple[PlanSpec | None, list[str], str]:
    """plan -> verify -> dedupe. Returns (plan, verification_warnings, source);
    plan is None when every registry variant has already been tried."""
    prepared = state.prepared
    cardinality = prepared.categorical_cardinality()
    tried = {a.signature for a in state.attempts}
    profile = {
        "primary_metric": prepared.primary_metric,
        "excluded_columns": prepared.excluded_columns,
        "balanced_class_weights": prepared.balanced_class_weights,
    }
    rejection: str | None = None
    duplicate_rejections = 0
    failed_plan: PlanSpec | None = None

    for attempt in range(MAX_VERIFICATION_RETRIES):
        set_task(f"Planning and verifying pipeline (attempt {attempt + 1}/{MAX_VERIFICATION_RETRIES})")
        verification_failure = state.last_verification_failure if failed_plan is not None else None
        plan = generate_plan(
            requirement,
            meta_features,
            similar_experiments,
            iteration=state.iteration + 1,
            previous_plan=failed_plan if verification_failure else state.previous_plan,
            reflection=state.last_reflection,
            verification_failure=verification_failure,
            attempts=state.attempts,
            replan=state.replan_requested,
            rejection=rejection,
            dataset_profile=profile,
        )
        verification = verify_plan(plan, requirement, state.df)
        if not verification.passed:
            logger.warning(
                "run=%s iteration=%s task=plan verification failed failures=%s",
                state.run_id, state.iteration + 1, verification.failures,
            )
            state.last_verification_failure = "; ".join(verification.failures)
            failed_plan = plan
            continue

        state.last_verification_failure = None
        failed_plan = None
        signature = plan_signature(plan, categorical_cardinality=cardinality)
        if signature not in tried:
            return plan, verification.warnings, "planner"
        if duplicate_rejections >= 1:
            fallback = _next_untried_plan(plan, requirement, tried, cardinality)
            if fallback is None:
                return None, verification.warnings, "exhausted"
            return fallback, verification.warnings, "deterministic_fallback"
        duplicate_rejections += 1
        rejection = (
            f"plan signature {signature} ({plan.model_family}) was already executed in this run. "
            f"Propose a configuration that is not in the 'Already tried' list."
        )

    if failed_plan is not None:
        raise RuntimeError(
            f"Plan repeatedly failed verification after {MAX_VERIFICATION_RETRIES} attempts: "
            f"{state.last_verification_failure}"
        )
    # Retries used up on duplicates: fall back deterministically.
    fallback = _next_untried_plan(plan, requirement, tried, cardinality)
    if fallback is None:
        return None, [], "exhausted"
    return fallback, [], "deterministic_fallback"


def _is_better(new: AttemptSummary, best: AttemptSummary | None, n_folds: int) -> bool:
    """Primary-metric CV mean decides; attempts within one standard error of
    each other are separated by the composite score (user priorities)."""
    if best is None:
        return True
    sign = 1.0 if greater_is_better(new.primary_metric) else -1.0
    diff = sign * (new.cv_mean - best.cv_mean)
    standard_error = max(new.cv_std, best.cv_std) / math.sqrt(max(n_folds, 1))
    if abs(diff) <= standard_error:
        return new.composite_score > best.composite_score
    return diff > 0


def _forced_stop(state: RunState, rationale: str, repo: ExperimentRepository) -> None:
    decision = DecisionOutput(decision="stop", rationale=rationale, forced=True)
    state.final_decision = decision
    if state.best is None:
        raise RuntimeError(f"{rationale} No attempt completed, so there is no model to keep.")
    _finalize(state, decision, repo)


def _run_one_iteration(state: RunState, repo: ExperimentRepository) -> None:
    requirement = state.requirement
    assert requirement is not None
    iteration_number = state.iteration + 1

    def set_task(task: str) -> None:
        state.current_task = task
        _record_activity(state, task, "plan" if "Plan" in task or "plan" in task else "agent")
        logger.info("run=%s iteration=%s task=%s", state.run_id, iteration_number, task)

    if state.cancel_event.is_set():
        _forced_stop(state, "Run cancelled by user; finalizing the best attempt so far.", repo)
        return
    if state.best is not None and state.budget_left_s() <= 0:
        _forced_stop(state, "Guardrail: the run's compute budget is exhausted; finalizing the best attempt so far.", repo)
        return

    prepared = _ensure_prepared(state)

    set_task("Finding similar experiments")
    meta_features = extract_meta_features(state.df, target_column=requirement.target_column)
    similar_experiments = retrieve_similar_experiments(
        meta_features, k=config.RETRIEVAL_TOP_K, repo=repo, problem_type=requirement.problem_type
    )

    plan, verification_warnings, plan_source = _obtain_plan(
        state, requirement, meta_features, similar_experiments, set_task
    )
    if plan is None:
        _forced_stop(state, "Guardrail: every registry configuration has already been tried.", repo)
        return

    # --- execute (CV on the training split only) ---
    set_task(f"Training and evaluating {plan.model_family}")

    def report_execution_progress(task: str) -> None:
        state.current_task = task
        _record_activity(state, task, "execute")
        logger.info("run=%s iteration=%s task=%s", state.run_id, iteration_number, task)

    budget = AttemptBudget(
        max_trials=config.HPO_MAX_TRIALS,
        hpo_timeout_s=max(1.0, min(config.HPO_TIMEOUT_S, state.budget_left_s() * 0.8)),
    )
    outcome = run_attempt(
        plan,
        prepared,
        budget=budget,
        progress_callback=report_execution_progress,
        cancel_event=state.cancel_event,
    )
    execution_result = outcome.result
    attempt = execution_result.attempt

    # --- validate ---
    set_task("Computing validation score")
    validation: ValidationReport = compute_validation(
        raw_metrics=execution_result.metrics,
        latency_ms_per_1k=attempt.resources.latency_ms_per_1k,
        model_family=attempt.model_family,
        problem_type=requirement.problem_type,
        weights=requirement.constraint_weights,
        per_block_diagnostics=execution_result.diagnostics,
        primary_metric=attempt.primary_metric,
    )
    state.score_history.append(validation.composite_score)

    summary = AttemptSummary(
        signature=outcome.signature,
        iteration=iteration_number,
        model_family=attempt.model_family,
        preprocessing=[s.component for s in plan.preprocessing],
        feature_engineering=[s.name for s in plan.feature_engineering],
        tuning=f"{plan.tuning.component}({min(plan.tuning.n_trials, config.HPO_MAX_TRIALS)})",
        primary_metric=attempt.primary_metric,
        cv_mean=attempt.cv_score,
        cv_std=attempt.cv[attempt.primary_metric].std,
        composite_score=validation.composite_score,
    )
    state.attempts.append(summary)
    if _is_better(summary, state.best.summary if state.best else None, attempt.split.n_folds):
        state.best = BestAttempt(iteration_number, plan, outcome, validation, summary)

    # --- reflect ---
    set_task("Reflecting on the weakest pipeline block")
    reflection = reflect(validation)
    state.weakest_block_history.append(reflection.weakest_block)

    # --- decide (deterministic guardrails live inside decide()) ---
    set_task("Choosing whether to refine, replan, or finish")
    state.iteration += 1
    if state.cancel_event.is_set():
        decision = DecisionOutput(decision="stop", rationale="Run cancelled by user.", forced=True)
    else:
        decision = decide(
            composite_score=validation.composite_score,
            score_history=state.score_history,
            weakest_block=reflection.weakest_block,
            weakest_block_history=state.weakest_block_history,
            iteration_count=state.iteration,
            max_iterations=config.MAX_ITERATIONS,
            budget_exhausted=state.budget_left_s() <= 0,
        )

    state.iteration_logs.append(
        IterationLog(
            iteration=state.iteration,
            plan_summary=_plan_summary(plan),
            composite_score=validation.composite_score,
            accuracy_norm=validation.accuracy_norm,
            latency_norm=validation.latency_norm,
            interp_norm=validation.interp_norm,
            weakest_block=reflection.weakest_block,
            decision=decision.decision,
            decision_rationale=decision.rationale,
            fallbacks=[*execution_result.fallbacks],
            verification_warnings=verification_warnings,
            signature=outcome.signature,
            plan_source=plan_source,
            primary_metric=attempt.primary_metric,
            cv_mean=summary.cv_mean,
            cv_std=summary.cv_std,
            latency_ms_per_1k=attempt.resources.latency_ms_per_1k,
            decision_forced=decision.forced,
            plan_adjustments=list(execution_result.plan_adjustments),
        )
    )

    state.previous_plan = plan
    state.last_reflection = reflection if decision.decision == "refine" else None
    state.replan_requested = decision.decision == "replan"
    state.final_decision = decision

    if decision.decision in ("accept", "stop"):
        set_task("Saving final model and report")
        _finalize(state, decision, repo)


def _assumptions(state: RunState) -> list[str]:
    assumptions = list(state.requirement.assumptions)
    if state.prepared is not None:
        assumptions.extend(state.prepared.notes)
        for column, reason in state.prepared.excluded_columns.items():
            assumptions.append(f"column '{column}' excluded from features ({reason})")
    coerced = state.load_report.get("numeric_coerced_columns") or []
    if coerced:
        assumptions.append(f"text columns converted to numbers: {', '.join(coerced)}")
    for log in state.iteration_logs:
        if log.plan_source == "deterministic_fallback":
            assumptions.append(f"iteration {log.iteration}: planner repeated a tried plan; used deterministic fallback")
        for fb in log.fallbacks:
            assumptions.append(f"iteration {log.iteration}: {fb['category']} '{fb['requested']}' unsupported -> used '{fb['used']}'")
        for w in log.verification_warnings:
            assumptions.append(f"iteration {log.iteration}: {w}")
        for note in log.plan_adjustments:
            assumptions.append(f"iteration {log.iteration}: {note}")
    return assumptions


def _finalize(state: RunState, decision: DecisionOutput, repo: ExperimentRepository) -> None:
    best = state.best
    assert best is not None and state.prepared is not None
    prepared = state.prepared
    attempt = best.outcome.result.attempt

    # The holdout is scored exactly once per run, for the selected attempt.
    holdout_metrics = holdout_evaluate(best.outcome.pipeline, prepared)
    state.holdout_evaluations += 1

    if should_generate_shap(state.requirement.constraint_weights, state.requirement.weight_confidence):
        state.current_task = "Generating SHAP feature importance"
        _record_activity(state, state.current_task, "execute")
        state.final_shap_summary, _ = shap_summary(
            best.outcome.pipeline, prepared.X_train, prepared.X_train, seed=prepared.seed
        )

    model_path = MODELS_DIR / f"{state.run_id}.joblib"
    joblib.dump(best.outcome.pipeline, model_path)
    state.model_path = str(model_path)
    state.final_metrics = holdout_metrics

    cv_metrics = {name: {"mean": s.mean, "std": s.std} for name, s in attempt.cv.items()}
    reproducibility = {
        "seed": prepared.seed,
        "code_version": provenance.code_version(),
        "lib_versions": provenance.lib_versions(),
        "dataset_fingerprint": state.dataset_fingerprint,
        "note": "LLM planning is not deterministic; given the recorded plan, execution is reproducible.",
    }
    state.final_report = {
        "run_id": state.run_id,
        "dataset_name": state.dataset_name,
        "target_column": state.requirement.target_column,
        "problem_type": state.requirement.problem_type,
        "model_family": attempt.model_family,
        "model_path": state.model_path,
        "primary_metric": attempt.primary_metric,
        "metrics": holdout_metrics,
        "holdout_metrics": holdout_metrics,
        "cv_metrics": cv_metrics,
        "composite_score": best.validation.composite_score,
        "best_iteration": best.iteration,
        "best_signature": best.summary.signature,
        "best_hyperparameters": attempt.best_params,
        "latency_ms_per_1k": attempt.resources.latency_ms_per_1k,
        "resources": attempt.resources.model_dump(),
        "constraint_weights": state.requirement.constraint_weights.model_dump(),
        "shap_summary": state.final_shap_summary,
        "assumptions": _assumptions(state),
        "data": {**prepared.summary(), "load_report": state.load_report},
        "final_decision": decision.decision,
        "final_decision_rationale": decision.rationale,
        "final_decision_forced": decision.forced,
        "run_history": [
            {
                "iteration": log.iteration,
                "plan_summary": log.plan_summary,
                "signature": log.signature,
                "plan_source": log.plan_source,
                "composite_score": log.composite_score,
                "primary_metric": log.primary_metric,
                "cv_mean": log.cv_mean,
                "cv_std": log.cv_std,
                "latency_ms_per_1k": log.latency_ms_per_1k,
                "weakest_block": log.weakest_block,
                "decision": log.decision,
                "decision_rationale": log.decision_rationale,
                "decision_forced": log.decision_forced,
            }
            for log in state.iteration_logs
        ],
        "reproducibility": reproducibility,
    }
    state.status = "finalized"

    repo.save_run(
        ExperimentRecord.new(
            run_id=state.run_id,
            dataset_name=state.dataset_name,
            problem_type=state.requirement.problem_type,
            target_column=state.requirement.target_column,
            meta_features=extract_meta_features(state.df, state.requirement.target_column).to_dict(),
            plan=best.plan.model_dump(),
            scores=state.score_history,
            weakest_block_history=state.weakest_block_history,
            final_decision=decision.decision,
            final_composite_score=best.validation.composite_score,
            status="finalized",
            iteration_logs=[asdict(log) for log in state.iteration_logs],
            final_metrics={"primary_metric": attempt.primary_metric, "cv": cv_metrics, "holdout": holdout_metrics},
            model_path=state.model_path,
            dataset_fingerprint=state.dataset_fingerprint,
            code_version=reproducibility["code_version"],
            lib_versions=reproducibility["lib_versions"],
            final_report=state.final_report,
        )
    )


def _persist_failure(state: RunState, repo: ExperimentRepository) -> None:
    """Failed runs are recorded too (audit G10), but never used as retrieval evidence."""
    requirement = state.requirement
    target = requirement.target_column if requirement else None
    try:
        meta = extract_meta_features(state.df, target if target in state.df.columns else None).to_dict()
    except Exception:  # never let failure bookkeeping mask the original error
        meta = {}
    repo.save_run(
        ExperimentRecord.new(
            run_id=state.run_id,
            dataset_name=state.dataset_name,
            problem_type=requirement.problem_type if requirement else None,
            target_column=target,
            meta_features=meta,
            plan=state.previous_plan.model_dump() if state.previous_plan else {},
            scores=state.score_history,
            weakest_block_history=state.weakest_block_history,
            final_decision=state.final_decision.decision if state.final_decision else None,
            final_composite_score=None,
            status="failed",
            error=state.error,
            iteration_logs=[asdict(log) for log in state.iteration_logs],
            dataset_fingerprint=state.dataset_fingerprint,
            code_version=provenance.code_version(),
            lib_versions=provenance.lib_versions(),
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
        state.current_task = "Run failed"
        state.error = str(exc)
        logger.exception("run=%s failed", state.run_id)
        try:
            _persist_failure(state, repo)
        except Exception:
            logger.exception("run=%s could not persist failure record", state.run_id)
        return state
    finally:
        if owns_repo:
            repo.close()
