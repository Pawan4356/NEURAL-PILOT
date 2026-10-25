"""Graph nodes. Each one wraps existing business logic and returns only the
state fields it changes.

LLM nodes: `understand`, `plan`, `reflect`. Everything else, including
`decide`, is deterministic (Phase 9: the routing rules are in `graph/decision.py`). The `clarify` node pauses the graph with `interrupt()`; on
resume LangGraph re-runs it from the top, so nothing before the
`interrupt()` call may have side effects.

Within an iteration, `resolve_plan` turns the planner's plan into registry
components and builds the challengers; `execute` measures every candidate
and keeps the best. The planner's plan is the one that is reflected on.

Errors are not caught here: `build.py` wraps every node so an exception is
recorded in state and routed to `fail`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from .. import config, reporting
from ..agents.planning_agent import generate_plan
from ..agents.reflection_agent import reflect as reflection_agent
from ..agents.requirement_agent import (
    InvalidClarification,
    apply_clarification_answer,
    match_column,
    needs_clarification,
    understand_requirement,
)
from ..candidates import WarmStart, build_candidates
from ..engine import AttemptBudget, plan_signature, run_attempt
from ..engine.explainability import shap_summary
from ..engine.runner import AttemptOutcome, holdout_evaluate
from ..meta_features import DatasetMetaFeatures, extract_meta_features
from ..plan_resolution import PlanProfile, PlanRejection, resolve
from ..knowledge.retrieve import check_citations, query_for, retrieve
from ..knowledge.sources import registry_component_names
from ..repository import SimilarExperiment
from ..reflection_guard import usable_suggestions
from ..schemas import (
    AttemptSummary,
    BestAttempt,
    CandidatePlan,
    DecisionOutput,
    IterationLog,
    PlanSpec,
    RequirementSpec,
    ValidationReport,
)
from ..scoring import compute_validation, should_generate_shap
from ..selection import is_better, next_untried_plan, plan_summary
from ..validation import evaluate_attempt, requires_glass_box
from ..verification import verify_plan
from . import resources
from .context import RunContext
from .decision import decide_route
from .state import NPState, plan_reset

logger = logging.getLogger(__name__)


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


def _plan_profile(state: NPState) -> PlanProfile:
    """What plan resolution needs to know about the dataset and the task."""
    requirement = state["requirement"]
    summary = state.get("data_summary") or {}
    return PlanProfile(
        problem_type=requirement.problem_type,
        n_rows=int(summary.get("n_train", summary.get("n_rows", 0))),
        categorical_cardinality=state.get("categorical_cardinality"),
        glass_box=requires_glass_box(requirement.constraints),
    )


def _memory_evidence(similar: list[SimilarExperiment]) -> dict[str, float]:
    """Best final score of each model family among similar past runs."""
    evidence: dict[str, float] = {}
    for item in similar:
        family = item.record.plan.get("model_family")
        score = item.record.final_composite_score
        if family and score is not None:
            evidence[family] = max(score, evidence.get(family, score))
    return evidence


def _memory_plans(similar: list[SimilarExperiment]) -> list[WarmStart]:
    """Plans of similar past runs, as warm-start proposals, each with the
    hyperparameters its model found. Records that do not parse against the
    current schema are skipped rather than failing the run."""
    proposals = []
    for item in similar:
        try:
            plan = PlanSpec.model_validate(item.record.plan)
        except ValidationError:
            continue
        report = item.record.final_report or {}
        params = report.get("best_hyperparameters") or {}
        proposals.append(WarmStart(plan=plan, best_params=params if report.get("model_family") == plan.model_family else {}))
    return proposals


def _input_columns(prepared) -> dict[str, str]:
    """Column names and types for the saved model's signature (no values)."""
    return {c: "string" if c in prepared.categorical_cols else "double" for c in prepared.X.columns}


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
    """Profile columns, fix the split, scan for leakage, open the tracking
    parent run — once per run."""
    ctx = runtime.context
    ctx.report("Profiling columns and checking for leakage", "execute")
    prepared = resources.prepared_data(state)
    df = resources.dataframe(state)
    requirement = state["requirement"]
    meta_features = extract_meta_features(df, target_column=requirement.target_column)
    parent_run = ctx.tracker.start_parent(
        state["run_id"],
        tags={
            "neuralpilot.dataset": state["dataset_name"],
            "neuralpilot.problem_type": requirement.problem_type,
            "neuralpilot.target": requirement.target_column,
        },
    )
    return {
        "data_summary": prepared.summary(),
        "data_notes": list(prepared.notes),
        "categorical_cardinality": prepared.categorical_cardinality(),
        "meta_features": asdict(meta_features),
        "budget_started_at": state.get("budget_started_at") or time.time(),
        "mlflow_run_id": parent_run,
    }


def retrieve_knowledge(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Similar past runs (usable evidence, same task type, above MIN_SIMILARITY)
    and the documentation passages the planner may cite (Phase 8)."""
    ctx = runtime.context
    ctx.report("Finding similar experiments and documentation")
    similar = ctx.repo.retrieve_similar_experiments(
        _meta_features(state), k=config.RETRIEVAL_TOP_K, problem_type=state["requirement"].problem_type
    )
    return {
        "similar_experiments": [{"run_id": s.record.run_id, "similarity": s.similarity} for s in similar],
        "knowledge_passages": _documentation(state, ctx),
    }


def _documentation(state: NPState, ctx: RunContext) -> list[dict]:
    """Passages from the offline index. Retrieval is advisory: if it fails the
    run continues without passages, and the failure is logged."""
    if ctx.knowledge is None:
        return []
    requirement = state["requirement"]
    try:
        query = query_for(_meta_features(state), requirement.problem_type, glass_box=_plan_profile(state).glass_box)
        passages = retrieve(
            ctx.knowledge,
            query,
            problem_type=requirement.problem_type,
            registry_components=registry_component_names(),
        )
    except Exception:
        logger.warning("run=%s documentation retrieval failed; continuing without it", state["run_id"], exc_info=True)
        return []
    return [passage.to_state() for passage in passages]


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
        unsupported=list(state.get("unsupported_recommendations") or []),
        passages=list(state.get("knowledge_passages") or []),
    )
    return {"candidate_plan": candidate, "plan_attempt": attempt}


def resolve_plan(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Verify the candidate, map its components onto the registry, and reject
    repeats of executed plans (audit G3).

    A plan that fails verification or cannot be resolved is sent back to the
    planner with the reason. A repeat is sent back once with an explicit
    rejection; after that (or when the attempts run out) the next untried
    registry variant is used instead. An accepted plan also gets its
    challengers built here (Phase 4)."""
    ctx = runtime.context
    candidate = state["candidate_plan"]
    requirement = state["requirement"]
    attempt = state.get("plan_attempt", 1)
    last_attempt = attempt >= config.MAX_PLAN_ATTEMPTS
    iteration = state.get("iteration", 0) + 1
    df = resources.dataframe(state)

    retrieved_ids = {p["id"] for p in state.get("knowledge_passages") or []}
    candidate, citation_warnings = check_citations(candidate, retrieved_ids)
    verification = verify_plan(candidate, requirement, df)
    if not verification.passed:
        failure = "; ".join(verification.failures)
        update = {"failed_plan": candidate, "verification_failure": failure}
        if last_attempt:
            update["error"] = (
                f"Plan repeatedly failed verification after {config.MAX_PLAN_ATTEMPTS} attempts: {failure}"
            )
        return update

    similar = _similar_experiments(state, ctx)
    resolution = resolve(
        candidate,
        _plan_profile(state),
        iteration=iteration,
        source="planner",
        prior_evidence=_memory_evidence(similar),
    )
    substitutions = list(resolution.substitutions)
    if isinstance(resolution, PlanRejection):
        failure = resolution.reason
        update = {
            "failed_plan": candidate,
            "verification_failure": failure,
            "unsupported_recommendations": substitutions,
        }
        if last_attempt:
            update["error"] = f"Plan could not be resolved after {config.MAX_PLAN_ATTEMPTS} attempts: {failure}"
        return update
    resolved = resolution.plan

    cardinality = state.get("categorical_cardinality")
    tried = {a.signature for a in state.get("attempts") or []}
    signature = plan_signature(resolved, categorical_cardinality=cardinality)
    base = {
        "failed_plan": None,
        "verification_failure": None,
        "verification_warnings": [*verification.warnings, *citation_warnings],
        "unsupported_recommendations": substitutions,
    }
    if signature not in tried:
        return {**base, **_accepted(resolved, "planner", state, similar, tried, iteration)}

    duplicate_rejections = state.get("duplicate_rejections", 0)
    if duplicate_rejections >= 1 or last_attempt:
        fallback = next_untried_plan(resolved, requirement, tried, cardinality)
        if fallback is None:
            return {**base, **forced_stop(state, "Guardrail: every registry configuration has already been tried.")}
        return {**base, **_accepted(fallback, "deterministic_fallback", state, similar, tried, iteration)}
    return {
        **base,
        "duplicate_rejections": duplicate_rejections + 1,
        "plan_rejection": (
            f"plan signature {signature} ({resolved.model_family}) was already executed in this run. "
            f"Propose a configuration that is not in the 'Already tried' list."
        ),
    }


def _accepted(
    plan: PlanSpec,
    source: str,
    state: NPState,
    similar: list[SimilarExperiment],
    tried: set[str],
    iteration: int,
) -> dict:
    """The plan to execute, plus the challengers that will run beside it."""
    candidates, substitutions = build_candidates(
        plan,
        _plan_profile(state),
        iteration=iteration,
        tried=tried,
        memory_plans=_memory_plans(similar),
        prior_evidence=_memory_evidence(similar),
    )
    return {
        "plan": plan,
        "plan_source": source,
        "candidates": candidates,
        "unsupported_recommendations": substitutions,
    }


def execute(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Measure every candidate of this iteration (CV on the training split
    only) and keep the best one so far.

    The planner's plan is always measured first and is the one that is
    reflected on. Challengers share the iteration's trial cap and timeout,
    are skipped when the budget is nearly used, and a failing challenger is
    recorded rather than failing the run.

    Scoring lives here rather than in a separate node because the fitted
    pipeline cannot be stored in graph state: it has to be kept (or
    dropped) in the same step that produced it."""
    ctx = runtime.context
    requirement = state["requirement"]
    iteration = state.get("iteration", 0) + 1
    cardinality = state.get("categorical_cardinality")
    candidates = state.get("candidates") or [CandidatePlan(source="planner", plan=state["plan"])]
    prepared = resources.prepared_data(state)

    share = len(candidates)
    trials_each = max(config.HPO_MAX_TRIALS // share, config.MIN_TRIALS_PER_CANDIDATE)
    timeout_each = max(1.0, min(config.HPO_TIMEOUT_S, budget_left_s(state) * 0.8)) / share

    best = state.get("best")
    best_summary = best.summary if best else None
    update: dict = {}
    attempts: list[AttemptSummary] = []
    log: list[dict] = []
    planner_result = None

    for index, candidate in enumerate(candidates):
        signature = plan_signature(candidate.plan, categorical_cardinality=cardinality)
        if index > 0:
            skip_reason = _challenger_skip_reason(state, ctx)
            if skip_reason:
                log.append(_candidate_entry(candidate, signature, "skipped", reason=skip_reason))
                continue
        label = candidate.plan.model_family if index == 0 else f"{candidate.plan.model_family} (challenger)"
        ctx.report(f"Training and evaluating {label}")
        try:
            outcome = run_attempt(
                candidate.plan,
                prepared,
                budget=AttemptBudget(max_trials=trials_each, hpo_timeout_s=timeout_each),
                progress_callback=lambda task: ctx.report(task, "execute"),
                cancel_event=ctx.cancel_event,
                tracker=ctx.tracker,
                tracking_tags={
                    "neuralpilot.iteration": str(iteration),
                    "neuralpilot.candidate_source": candidate.source,
                },
                warm_start_params=candidate.warm_start_params or None,
            )
        except Exception as exc:
            if index == 0:
                raise
            logger.warning("run=%s challenger %s failed", state["run_id"], candidate.plan.model_family, exc_info=True)
            log.append(_candidate_entry(candidate, signature, "failed", reason=str(exc) or type(exc).__name__))
            continue

        validation, summary = _score(outcome, candidate, iteration, requirement)
        attempts.append(summary)
        log.append(_candidate_entry(candidate, signature, "ok", summary=summary))
        if index == 0:
            planner_result = (outcome.result, validation, summary)
        if is_better(summary, best_summary, outcome.result.attempt.split.n_folds):
            resources.keep_best_pipeline(state["run_id"], outcome.pipeline)
            best = BestAttempt(
                iteration=iteration,
                plan=candidate.plan,
                summary=summary,
                validation=validation,
                attempt=outcome.result.attempt,
            )
            best_summary = summary
            update["best"] = best

    execution, validation, summary = planner_result
    update.update(
        {
            "last_execution": execution,
            "last_validation": validation,
            "last_summary": summary,
            "attempts": attempts,
            "score_history": [validation.composite_score],
            "iteration_candidates": log,
        }
    )
    return update


def _score(
    outcome: AttemptOutcome, candidate: CandidatePlan, iteration: int, requirement: RequirementSpec
) -> tuple[ValidationReport, AttemptSummary]:
    """Composite score, constraint checks and the compact attempt record."""
    execution = outcome.result
    attempt = execution.attempt
    validation = compute_validation(
        raw_metrics=execution.metrics,
        latency_ms_per_1k=attempt.resources.latency_ms_per_1k,
        model_family=attempt.model_family,
        problem_type=requirement.problem_type,
        weights=requirement.constraint_weights,
        per_block_diagnostics=execution.diagnostics,
        primary_metric=attempt.primary_metric,
    )
    validation = evaluate_attempt(validation, attempt, requirement)
    plan_spec = candidate.plan
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
        source=candidate.source,
        latency_ms_per_1k=attempt.resources.latency_ms_per_1k,
        interp_norm=validation.interp_norm,
        feasible=validation.feasible,
        constraint_results=validation.constraint_results,
        best_params=attempt.best_params,
    )
    return validation, summary


def _challenger_skip_reason(state: NPState, ctx: RunContext) -> str | None:
    if ctx.cancelled():
        return "run cancelled"
    if budget_left_s(state) < config.CHALLENGER_MIN_BUDGET_S:
        return "compute budget nearly used"
    return None


def _candidate_entry(
    candidate: CandidatePlan,
    signature: str,
    status: str,
    *,
    summary: AttemptSummary | None = None,
    reason: str | None = None,
) -> dict:
    """One row of the iteration's candidate log (Deliverable 8.5)."""
    return {
        "signature": signature,
        "model_family": candidate.plan.model_family,
        "source": candidate.source,
        "status": status,
        "cv_mean": summary.cv_mean if summary else None,
        "cv_std": summary.cv_std if summary else None,
        "feasible": summary.feasible if summary else None,
        "reason": reason,
    }


def reflect(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Ask the Reflection Agent, then keep only the suggestions this run can use:
    each must resolve against the registry to a configuration not yet executed.
    A reflection whose suggestions are all dead proposes no change."""
    ctx = runtime.context
    ctx.report("Reflecting on the weakest pipeline block")
    iteration = state.get("iteration", 0) + 1
    attempts = list(state.get("attempts") or [])
    reflection = reflection_agent(
        state["last_validation"],
        client=ctx.llm,
        attempts=attempts,
        iteration=iteration,
        budget_left_s=budget_left_s(state),
        unsupported=list(state.get("unsupported_recommendations") or []),
    )

    tried = {a.signature for a in attempts}
    usable, substitutions = usable_suggestions(
        reflection.suggestions, state["plan"], _plan_profile(state), iteration=iteration, tried=tried
    )
    kept = [suggestion for suggestion, _ in usable]
    notes = list(reflection.guard_notes)
    if len(kept) < len(reflection.suggestions):
        notes.append(
            f"{len(reflection.suggestions) - len(kept)} suggested component(s) were unsupported or already executed"
        )
    proposed = reflection.proposed_action
    if reflection.suggestions and not kept:
        proposed = "none"
    reflection = reflection.model_copy(
        update={"suggestions": kept, "guard_notes": notes, "proposed_action": proposed}
    )
    return {
        "reflection": reflection,
        "weakest_block_history": [reflection.weakest_block],
        "unsupported_recommendations": substitutions,
    }


def decide(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """accept / refine / replan / stop, by the deterministic rules in
    `graph/decision.py`. No LLM is called here: every input is numeric or a
    checked claim."""
    ctx = runtime.context
    ctx.report("Choosing whether to refine, replan, or finish")
    iteration = state.get("iteration", 0) + 1
    validation = state["last_validation"]
    reflection = state["reflection"]
    execution = state["last_execution"]
    attempt = execution.attempt
    summary = state["last_summary"]
    best = state.get("best")

    decision = decide_route(
        cancelled=ctx.cancelled(),
        budget_exhausted=budget_left_s(state) <= 0,
        iteration=iteration,
        max_iterations=state.get("max_iterations", config.MAX_ITERATIONS),
        best_feasible=best is not None and best.validation.feasible,
        score_history=list(state.get("score_history") or []),
        weakest_block_history=list(state.get("weakest_block_history") or []),
        proposed_action=reflection.proposed_action,
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
        candidates=list(state.get("iteration_candidates") or []),
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
    ctx.tracker.log_final(
        pipeline=pipeline,
        input_columns=_input_columns(prepared),
        holdout=holdout_metrics,
        report=report,
    )
    return {
        "holdout_evaluations": state.get("holdout_evaluations", 0) + 1,
        "model_path": model_path,
        "final_metrics": holdout_metrics,
        "final_report": report,
    }


def persist(state: NPState, runtime: Runtime[RunContext]) -> dict:
    """Write the finalized run to experiment memory."""
    ctx = runtime.context
    df = resources.dataframe(state)
    meta_features = extract_meta_features(df, state["requirement"].target_column).to_dict()
    ctx.repo.save_run(reporting.finalized_record(state, meta_features))
    ctx.repo.save_attempts(state["run_id"], list(state.get("attempts") or []))
    ctx.repo.save_unsupported(state["run_id"], list(state.get("unsupported_recommendations") or []))
    ctx.tracker.end("finalized")
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
    ctx.repo.save_attempts(state["run_id"], list(state.get("attempts") or []))
    ctx.repo.save_unsupported(state["run_id"], list(state.get("unsupported_recommendations") or []))
    ctx.tracker.end("failed")
    resources.release_run(state["run_id"])
    return {"status": "failed", "error": error}
