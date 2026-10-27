"""Final report and experiment records, built from graph state.

Kept separate from the graph nodes so the report's shape can be read (and
changed) in one place. Everything here is pure: it turns state into dicts
and `ExperimentRecord`s; the nodes decide when to call it.
"""

from __future__ import annotations

from . import provenance
from .repository import ExperimentRecord
from .schemas import DecisionOutput
from .validation.report import summarize_run


def collect_assumptions(state: dict) -> list[str]:
    """Every default, fallback and exclusion the run relied on, in plain words."""
    requirement = state.get("requirement")
    assumptions = list(requirement.assumptions) if requirement else []
    assumptions.extend(state.get("data_notes") or [])
    excluded = (state.get("data_summary") or {}).get("excluded_columns", {})
    for column, reason in excluded.items():
        assumptions.append(f"column '{column}' excluded from features ({reason})")
    coerced = (state.get("load_report") or {}).get("numeric_coerced_columns") or []
    if coerced:
        assumptions.append(f"text columns converted to numbers: {', '.join(coerced)}")
    for log in state.get("iteration_logs") or []:
        if log.plan_source == "deterministic_fallback":
            assumptions.append(f"iteration {log.iteration}: planner repeated a tried plan; used deterministic fallback")
        for fb in log.fallbacks:
            assumptions.append(f"iteration {log.iteration}: {fb['category']} '{fb['requested']}' unsupported -> used '{fb['used']}'")
        for warning in log.verification_warnings:
            assumptions.append(f"iteration {log.iteration}: {warning}")
        for note in log.plan_adjustments:
            assumptions.append(f"iteration {log.iteration}: {note}")
    for item in state.get("unsupported_recommendations") or []:
        change = " (semantic change)" if item.semantic_change else ""
        assumptions.append(
            f"iteration {item.iteration}: {item.category} '{item.requested}' not used ({item.reason.replace('_', ' ')})"
            f" -> '{item.alternative_used}'{change}"
        )
    return assumptions


def reproducibility_block(state: dict) -> dict:
    return {
        "seed": state.get("seed"),
        "code_version": provenance.code_version(),
        "lib_versions": provenance.lib_versions(),
        "dataset_fingerprint": state.get("dataset_fingerprint"),
        "note": "LLM planning is not deterministic; given the recorded plan, execution is reproducible.",
    }


def cv_metrics_of(state: dict) -> dict:
    attempt = state["best"].attempt
    return {name: {"mean": s.mean, "std": s.std} for name, s in attempt.cv.items()}


def validation_block(state: dict) -> dict:
    """Phase 5: whether the chosen model meets the user's hard constraints,
    and the run-level view (feasible count, Pareto front, closest attempts)."""
    best = state["best"]
    summary = summarize_run(list(state.get("attempts") or []), best.summary)
    return {
        "acceptable": best.validation.feasible,
        "feasible": best.validation.feasible,
        "constraint_table": [r.model_dump() for r in best.validation.constraint_results],
        "feasible_count": summary["feasible_count"],
        "attempt_count": summary["attempt_count"],
        "pareto_front": summary["pareto_front"],
        "closest_per_violation": summary["closest_per_violation"],
    }


def tuning_block(state: dict) -> dict:
    """What the tuner managed to do: a plan's trial count is an upper bound,
    not a promise, because the attempt's timeout can cut the search short."""
    for block in state["best"].validation.per_block_diagnostics:
        if block.block == "hyperparameters":
            details = block.details or {}
            return {
                "tuner": details.get("tuner"),
                "requested_trials": details.get("n_trials"),
                "completed_trials": details.get("n_trials_completed"),
                "pruned_trials": details.get("n_trials_pruned"),
                "failed_trials": details.get("n_trials_failed"),
                "stopped_reason": details.get("stopped_reason"),
            }
    return {}


def build_final_report(
    state: dict,
    *,
    decision: DecisionOutput,
    holdout_metrics: dict,
    model_path: str,
    shap_summary: dict | None,
    reproducibility: dict,
) -> dict:
    best = state["best"]
    attempt = best.attempt
    requirement = state["requirement"]
    return {
        "run_id": state["run_id"],
        "dataset_name": state["dataset_name"],
        "target_column": requirement.target_column,
        "problem_type": requirement.problem_type,
        "model_family": attempt.model_family,
        "model_path": model_path,
        "primary_metric": attempt.primary_metric,
        "metrics": holdout_metrics,
        "holdout_metrics": holdout_metrics,
        "cv_metrics": cv_metrics_of(state),
        "composite_score": best.validation.composite_score,
        "best_iteration": best.iteration,
        "best_signature": best.summary.signature,
        "best_hyperparameters": attempt.best_params,
        "latency_ms_per_1k": attempt.resources.latency_ms_per_1k,
        "resources": attempt.resources.model_dump(),
        "tuning": tuning_block(state),
        "constraint_weights": requirement.constraint_weights.model_dump(),
        **validation_block(state),
        "best_plan": best.plan.model_dump(),
        "unsupported_recommendations": [u.model_dump() for u in state.get("unsupported_recommendations") or []],
        "mlflow_run_id": state.get("mlflow_run_id"),
        "shap_summary": shap_summary,
        "assumptions": collect_assumptions(state),
        "data": {**(state.get("data_summary") or {}), "load_report": state.get("load_report") or {}},
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
                "candidates": log.candidates,
            }
            for log in state.get("iteration_logs") or []
        ],
        "reproducibility": reproducibility,
    }


def finalized_record(state: dict, meta_features: dict) -> ExperimentRecord:
    best = state["best"]
    report = state["final_report"]
    requirement = state["requirement"]
    return ExperimentRecord.new(
        run_id=state["run_id"],
        dataset_name=state["dataset_name"],
        problem_type=requirement.problem_type,
        target_column=requirement.target_column,
        meta_features=meta_features,
        plan=best.plan.model_dump(),
        scores=list(state.get("score_history") or []),
        weakest_block_history=list(state.get("weakest_block_history") or []),
        final_decision=state["decision"].decision,
        final_composite_score=best.validation.composite_score,
        status="finalized",
        iteration_logs=[log.model_dump() for log in state.get("iteration_logs") or []],
        final_metrics={
            "primary_metric": best.attempt.primary_metric,
            "cv": report["cv_metrics"],
            "holdout": report["holdout_metrics"],
        },
        model_path=state["model_path"],
        dataset_fingerprint=state.get("dataset_fingerprint"),
        code_version=report["reproducibility"]["code_version"],
        lib_versions=report["reproducibility"]["lib_versions"],
        final_report=report,
        mlflow_run_id=state.get("mlflow_run_id"),
    )


def failed_record(state: dict, meta_features: dict) -> ExperimentRecord:
    """Failed runs are recorded too (audit G10), but never used as retrieval evidence."""
    requirement = state.get("requirement")
    previous_plan = state.get("previous_plan")
    decision = state.get("decision")
    return ExperimentRecord.new(
        run_id=state["run_id"],
        dataset_name=state["dataset_name"],
        problem_type=requirement.problem_type if requirement else None,
        target_column=requirement.target_column if requirement else None,
        meta_features=meta_features,
        plan=previous_plan.model_dump() if previous_plan else {},
        scores=list(state.get("score_history") or []),
        weakest_block_history=list(state.get("weakest_block_history") or []),
        final_decision=decision.decision if decision else None,
        final_composite_score=None,
        status="failed",
        error=state.get("error"),
        iteration_logs=[log.model_dump() for log in state.get("iteration_logs") or []],
        dataset_fingerprint=state.get("dataset_fingerprint"),
        code_version=provenance.code_version(),
        lib_versions=provenance.lib_versions(),
        mlflow_run_id=state.get("mlflow_run_id"),
    )
