"""Graph nodes. Each one wraps existing business logic and returns only the
state fields it changes.

LLM nodes: `understand`, `plan`, `reflect`, `decide` (the decision agent's
guardrails short-circuit before any LLM call). Everything else is
deterministic. The `clarify` node pauses the graph with `interrupt()`; on
resume LangGraph re-runs it from the top, so nothing before the
`interrupt()` call may have side effects.

Errors are not caught here: `build.py` wraps every node so an exception is
recorded in state and routed to `fail`.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from .. import config, reporting
from ..agents.decision_agent import decide as decision_agent
from ..agents.planning_agent import generate_plan
from ..agents.reflection_agent import reflect as reflection_agent
from ..agents.requirement_agent import (
    InvalidClarification,
    apply_clarification_answer,
    match_column,
    needs_clarification,
    understand_requirement,
)
from ..engine import AttemptBudget, plan_signature, run_attempt
from ..engine.explainability import shap_summary
from ..engine.runner import holdout_evaluate
from ..meta_features import DatasetMetaFeatures, extract_meta_features
from ..repository import SimilarExperiment
from ..schemas import AttemptSummary, BestAttempt, DecisionOutput, IterationLog
from ..scoring import compute_validation, should_generate_shap
from ..selection import is_better, next_untried_plan, plan_summary
from ..verification import verify_plan
from . import resources
from .context import RunContext
from .state import NPState, plan_reset


# --- helpers -----------------------------------------------------------------


def budget_left_s(state: NPState) -> float:
    started = state.get("budget_started_at")
    if started is None:
        return state["budget_s"]
    return state["budget_s"] - (time.time() - started)


def forced_stop(state: NPState, rationale: str) -> dict:
    """Stop now and keep the best attempt; without one, the run fails."""
    update: dict = {"decision": DecisionOutput(decision="stop", rationale=rationale, forced=True)}
    if state.get("best") is None:
        update["error"] = f"{rationale} No attempt completed, so there is no model to keep."
    return update


def _meta_features(state: NPState) -> DatasetMetaFeatures:
    return DatasetMetaFeatures(**state["meta_features"])


def _similar_experiments(state: NPState, ctx: RunContext) -> list[SimilarExperiment]:
    similar = []
    for item in state.get("similar_experiments") or []:
        record = ctx.repo.get_run(item["run_id"])
        if record is not None:
            similar.append(SimilarExperiment(record, item["similarity"]))
    return similar


def _ask_for_target(spec, columns: list[str], reason: str) -> dict:
    return {
        "requirement": spec,
        "status": "awaiting_clarification",
        "clarification_question": (
            f"{reason} Which column should be the prediction target? Available columns: {', '.join(columns)}"
        ),
    }


# --- requirement understanding ---------------------------------------------


def understand(state: NPState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    ctx.report("Understanding requirements and inspecting dataset", "requirement")
    df = resources.dataframe(state)
    meta_features = extract_meta_features(df, target_column=None)
    spec = understand_requirement(state["user_goal"], meta_features, client=ctx.llm)

    columns = [str(c) for c in df.columns]
    matched = match_column(spec.target_column, columns)
    if matched is None:
        return _ask_for_target(spec, columns, f"I couldn't find a column named '{spec.target_column}'.")
    spec.target_column = matched

    if needs_clarification(spec):
        return _ask_for_target(
            spec,
            columns,
            f"I'm not fully sure which column you want to predict "
            f"(best guess: '{spec.target_column}', confidence {spec.target_confidence:.0%}).",
        )
    return {"requirement": spec, "status": "running", "clarification_question": None}


def clarify(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Pause until the user names the target column (§7)."""
    df = resources.dataframe(state)
    columns = [str(c) for c in df.columns]
    answer = interrupt({"question": state["clarification_question"], "columns": columns})

    rounds = state.get("clarification_rounds", 0) + 1
    try:
        requirement = apply_clarification_answer(state["requirement"], str(answer), df)
    except InvalidClarification as exc:
        update = {"clarification_rounds": rounds, "clarification_question": f"{exc}. Which column should be the prediction target?"}
        if rounds >= config.MAX_CLARIFICATION_ROUNDS:
            update["error"] = f"Could not determine the target column: {exc}"
        return update

    runtime.context.report("Resuming after target column clarification", "requirement")
    return {
        "requirement": requirement,
        "clarification_rounds": rounds,
        "clarification_question": None,
        "status": "running",
    }


# --- data and knowledge ------------------------------------------------------


def prepare(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Profile columns, fix the split, scan for leakage — once per run."""
    ctx = runtime.context
    ctx.report("Profiling columns and checking for leakage", "execute")
    prepared = resources.prepared_data(state)
    df = resources.dataframe(state)
    meta_features = extract_meta_features(df, target_column=state["requirement"].target_column)
    return {
        "data_summary": prepared.summary(),
        "data_notes": list(prepared.notes),
        "categorical_cardinality": prepared.categorical_cardinality(),
        "meta_features": asdict(meta_features),
        "budget_started_at": state.get("budget_started_at") or time.time(),
    }


def retrieve_knowledge(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Similar past runs (finalized, same task type, above MIN_SIMILARITY)."""
    ctx = runtime.context
    ctx.report("Finding similar experiments")
    similar = ctx.repo.retrieve_similar_experiments(
        _meta_features(state), k=config.RETRIEVAL_TOP_K, problem_type=state["requirement"].problem_type
    )
    return {"similar_experiments": [{"run_id": s.record.run_id, "similarity": s.similarity} for s in similar]}


# --- one iteration -----------------------------------------------------------


def start_iteration(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Guardrails before spending more compute: cancellation and budget."""
    ctx = runtime.context
    if ctx.cancelled():
        return forced_stop(state, "Run cancelled by user; finalizing the best attempt so far.")
    if state.get("best") is not None and budget_left_s(state) <= 0:
        return forced_stop(
            state, "Guardrail: the run's compute budget is exhausted; finalizing the best attempt so far."
        )
    return plan_reset()


def plan(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Ask the planning agent for one pipeline plan."""
    ctx = runtime.context
    attempt = state.get("plan_attempt", 0) + 1
    ctx.report(f"Planning and verifying pipeline (attempt {attempt}/{config.MAX_PLAN_ATTEMPTS})", "plan")

    failed_plan = state.get("failed_plan")
    verification_failure = state.get("verification_failure") if failed_plan is not None else None
    data_summary = state["data_summary"]
    candidate = generate_plan(
        state["requirement"],
        _meta_features(state),
        _similar_experiments(state, ctx),
        client=ctx.llm,
        iteration=state.get("iteration", 0) + 1,
        previous_plan=failed_plan if verification_failure else state.get("previous_plan"),
        reflection=state.get("last_reflection"),
        verification_failure=verification_failure,
        attempts=list(state.get("attempts") or []),
        replan=state.get("replan_requested", False),
        rejection=state.get("plan_rejection"),
        dataset_profile={
            "primary_metric": data_summary["primary_metric"],
            "excluded_columns": data_summary["excluded_columns"],
            "balanced_class_weights": data_summary["balanced_class_weights"],
        },
    )
    return {"candidate_plan": candidate, "plan_attempt": attempt}


def resolve_plan(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Verify the candidate and reject repeats of executed plans (audit G3).

    A plan that fails verification is sent back to the planner with the
    reason. A repeat is sent back once with an explicit rejection; after
    that (or when the attempts run out) the next untried registry variant
    is used instead."""
    candidate = state["candidate_plan"]
    requirement = state["requirement"]
    attempt = state.get("plan_attempt", 1)
    last_attempt = attempt >= config.MAX_PLAN_ATTEMPTS
    df = resources.dataframe(state)

    verification = verify_plan(candidate, requirement, df)
    if not verification.passed:
        failure = "; ".join(verification.failures)
        update = {"failed_plan": candidate, "verification_failure": failure}
        if last_attempt:
            update["error"] = (
                f"Plan repeatedly failed verification after {config.MAX_PLAN_ATTEMPTS} attempts: {failure}"
            )
        return update

    cardinality = state.get("categorical_cardinality")
    tried = {a.signature for a in state.get("attempts") or []}
    signature = plan_signature(candidate, categorical_cardinality=cardinality)
    base = {"failed_plan": None, "verification_failure": None, "verification_warnings": verification.warnings}
    if signature not in tried:
        return {**base, "plan": candidate, "plan_source": "planner"}

    duplicate_rejections = state.get("duplicate_rejections", 0)
    if duplicate_rejections >= 1 or last_attempt:
        fallback = next_untried_plan(candidate, requirement, tried, cardinality)
        if fallback is None:
            return {**base, **forced_stop(state, "Guardrail: every registry configuration has already been tried.")}
        return {**base, "plan": fallback, "plan_source": "deterministic_fallback"}
    return {
        **base,
        "duplicate_rejections": duplicate_rejections + 1,
        "plan_rejection": (
            f"plan signature {signature} ({candidate.model_family}) was already executed in this run. "
            f"Propose a configuration that is not in the 'Already tried' list."
        ),
    }


def execute(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Run one attempt (CV on the training split only), score it, and keep
    it if it is the best so far.

    Scoring lives here rather than in a separate node because the fitted
    pipeline cannot be stored in graph state: it has to be kept (or
    dropped) in the same step that produced it."""
    ctx = runtime.context
    plan_spec = state["plan"]
    requirement = state["requirement"]
    iteration = state.get("iteration", 0) + 1
    ctx.report(f"Training and evaluating {plan_spec.model_family}")

    prepared = resources.prepared_data(state)
    budget = AttemptBudget(
        max_trials=config.HPO_MAX_TRIALS,
        hpo_timeout_s=max(1.0, min(config.HPO_TIMEOUT_S, budget_left_s(state) * 0.8)),
    )
    outcome = run_attempt(
        plan_spec,
        prepared,
        budget=budget,
        progress_callback=lambda task: ctx.report(task, "execute"),
        cancel_event=ctx.cancel_event,
    )
    execution = outcome.result
    attempt = execution.attempt

    ctx.report("Computing validation score")
    validation = compute_validation(
        raw_metrics=execution.metrics,
        latency_ms_per_1k=attempt.resources.latency_ms_per_1k,
        model_family=attempt.model_family,
        problem_type=requirement.problem_type,
        weights=requirement.constraint_weights,
        per_block_diagnostics=execution.diagnostics,
        primary_metric=attempt.primary_metric,
    )
    summary = AttemptSummary(
        signature=outcome.signature,
        iteration=iteration,
        model_family=attempt.model_family,
        preprocessing=[s.component for s in plan_spec.preprocessing],
        feature_engineering=[s.name for s in plan_spec.feature_engineering],
        tuning=f"{plan_spec.tuning.component}({min(plan_spec.tuning.n_trials, config.HPO_MAX_TRIALS)})",
        primary_metric=attempt.primary_metric,
        cv_mean=attempt.cv_score,
        cv_std=attempt.cv[attempt.primary_metric].std,
        composite_score=validation.composite_score,
    )

    update: dict = {
        "last_execution": execution,
        "last_validation": validation,
        "attempts": [summary],
        "score_history": [validation.composite_score],
    }
    current_best = state.get("best")
    if is_better(summary, current_best.summary if current_best else None, attempt.split.n_folds):
        resources.keep_best_pipeline(state["run_id"], outcome.pipeline)
        update["best"] = BestAttempt(
            iteration=iteration, plan=plan_spec, summary=summary, validation=validation, attempt=attempt
        )
    return update


def reflect(state: NPState, runtime: Runtime[RunContext]) -> dict:
    runtime.context.report("Reflecting on the weakest pipeline block")
    reflection = reflection_agent(state["last_validation"], client=runtime.context.llm)
    return {"reflection": reflection, "weakest_block_history": [reflection.weakest_block]}


def decide(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """accept / refine / replan / stop. Guardrails (budget, iteration cap,
    plateau, cancellation) force "stop" without asking the LLM."""
    ctx = runtime.context
    ctx.report("Choosing whether to refine, replan, or finish")
    iteration = state.get("iteration", 0) + 1
    validation = state["last_validation"]
    reflection = state["reflection"]
    execution = state["last_execution"]
    attempt = execution.attempt
    summary = state["attempts"][-1]

    if ctx.cancelled():
        decision = DecisionOutput(decision="stop", rationale="Run cancelled by user.", forced=True)
    else:
        decision = decision_agent(
            composite_score=validation.composite_score,
            score_history=list(state.get("score_history") or []),
            weakest_block=reflection.weakest_block,
            weakest_block_history=list(state.get("weakest_block_history") or []),
            iteration_count=iteration,
            max_iterations=state.get("max_iterations", config.MAX_ITERATIONS),
            budget_exhausted=budget_left_s(state) <= 0,
            client=ctx.llm,
        )

    log = IterationLog(
        iteration=iteration,
        plan_summary=plan_summary(state["plan"]),
        composite_score=validation.composite_score,
        accuracy_norm=validation.accuracy_norm,
        latency_norm=validation.latency_norm,
        interp_norm=validation.interp_norm,
        weakest_block=reflection.weakest_block,
        decision=decision.decision,
        decision_rationale=decision.rationale,
        fallbacks=list(execution.fallbacks),
        verification_warnings=list(state.get("verification_warnings") or []),
        signature=summary.signature,
        plan_source=state.get("plan_source", "planner"),
        primary_metric=attempt.primary_metric,
        cv_mean=summary.cv_mean,
        cv_std=summary.cv_std,
        latency_ms_per_1k=attempt.resources.latency_ms_per_1k,
        decision_forced=decision.forced,
        plan_adjustments=list(execution.plan_adjustments),
    )
    return {
        "iteration": iteration,
        "iteration_logs": [log],
        "decision": decision,
        "previous_plan": state["plan"],
        "last_reflection": reflection if decision.decision == "refine" else None,
        "replan_requested": decision.decision == "replan",
    }


# --- end of run --------------------------------------------------------------


def finalize(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Score the best attempt — not merely the last — once on the holdout,
    explain it, and build the report (audit G1, G7)."""
    ctx = runtime.context
    ctx.report("Saving final model and report")
    run_id = state["run_id"]
    requirement = state["requirement"]
    prepared = resources.prepared_data(state)
    pipeline = resources.best_pipeline(run_id)

    holdout_metrics = holdout_evaluate(pipeline, prepared)

    shap = None
    if should_generate_shap(requirement.constraint_weights, requirement.weight_confidence):
        ctx.report("Generating SHAP feature importance", "execute")
        shap, _ = shap_summary(pipeline, prepared.X_train, prepared.X_train, seed=prepared.seed)

    model_path = str(resources.model_path(run_id))
    report = reporting.build_final_report(
        state,
        decision=state["decision"],
        holdout_metrics=holdout_metrics,
        model_path=model_path,
        shap_summary=shap,
        reproducibility=reporting.reproducibility_block(state),
    )
    return {
        "holdout_evaluations": state.get("holdout_evaluations", 0) + 1,
        "model_path": model_path,
        "final_metrics": holdout_metrics,
        "final_report": report,
    }


def persist(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Write the finalized run to experiment memory."""
    df = resources.dataframe(state)
    meta_features = extract_meta_features(df, state["requirement"].target_column).to_dict()
    runtime.context.repo.save_run(reporting.finalized_record(state, meta_features))
    resources.release_run(state["run_id"])
    return {"status": "finalized"}


def fail(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Record the failure (audit G10). Bookkeeping errors here are logged
    by the wrapper but never mask the original error."""
    ctx = runtime.context
    error = state.get("error") or "The run stopped without a usable plan."
    ctx.report("Run failed", "agent")
    failed_state = {**state, "error": error}
    meta_features: dict = {}
    try:
        df = resources.dataframe(state)
        requirement = state.get("requirement")
        target = requirement.target_column if requirement else None
        meta_features = extract_meta_features(df, target if target in df.columns else None).to_dict()
    except Exception:
        pass
    ctx.repo.save_run(reporting.failed_record(failed_state, meta_features))
    resources.release_run(state["run_id"])
    return {"status": "failed", "error": error}
