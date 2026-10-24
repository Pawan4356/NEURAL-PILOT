"""Validation report builders (Phase 5, Deliverable 9.3-9.4).

Two levels:
- per attempt: `evaluate_attempt` attaches the constraint table and the
  feasibility flag to the attempt's `ValidationReport`;
- per run: `summarize_run` gives the feasible count, the Pareto front over
  (primary metric, latency, interpretability), and, when nothing is
  feasible, the closest attempt for each violated constraint.

The Pareto front and the closest attempts are shown to the user; they do not
choose the model. Selection is `selection.is_better`: feasible first, then
the primary metric.
"""

from __future__ import annotations

from ..engine.evaluation import greater_is_better
from ..registry import get_interp_class
from ..schemas import AttemptResult, AttemptSummary, ConstraintResult, RequirementSpec, ValidationReport
from .constraints import check_constraints, is_feasible


def evaluate_attempt(
    validation: ValidationReport, attempt: AttemptResult, requirement: RequirementSpec
) -> ValidationReport:
    results = check_constraints(
        requirement.constraints, attempt, interp_class=get_interp_class(attempt.model_family)
    )
    return validation.model_copy(update={"constraint_results": results, "feasible": is_feasible(results)})


def pareto_front(attempts: list[AttemptSummary]) -> list[str]:
    """Signatures of attempts not beaten on every objective by another attempt.

    Objectives: primary metric (higher is better, sign-adjusted), latency
    (lower is better) and interpretability (higher is better)."""
    points = {a.signature: _objectives(a) for a in attempts}
    front = []
    for signature, point in points.items():
        dominated = any(
            _dominates(other, point) for other_sig, other in points.items() if other_sig != signature
        )
        if not dominated:
            front.append(signature)
    return front


def closest_per_violation(best: AttemptSummary | None, attempts: list[AttemptSummary]) -> dict[str, dict]:
    """For each constraint the best attempt violates, the attempt that came
    nearest to meeting it, so the user can see what a looser limit would cost."""
    if best is None:
        return {}
    closest: dict[str, dict] = {}
    for violated in (r for r in best.constraint_results if not r.passed):
        ranked = sorted(attempts, key=lambda a: (_gap(_result(a, violated.name)), -a.cv_mean))
        if not ranked:
            continue
        winner = ranked[0]
        result = _result(winner, violated.name)
        closest[violated.name] = {
            "signature": winner.signature,
            "model_family": winner.model_family,
            "measured": result.measured if result else None,
            "threshold": violated.threshold,
            "passed": bool(result and result.passed),
        }
    return closest


def summarize_run(attempts: list[AttemptSummary], best: AttemptSummary | None) -> dict:
    return {
        "attempt_count": len(attempts),
        "feasible_count": sum(1 for a in attempts if a.feasible),
        "pareto_front": pareto_front(attempts),
        "best_feasible": bool(best and best.feasible),
        "constraint_table": [r.model_dump() for r in best.constraint_results] if best else [],
        "closest_per_violation": closest_per_violation(best, attempts),
    }


def _objectives(attempt: AttemptSummary) -> tuple[float, float, float]:
    metric = attempt.cv_mean if greater_is_better(attempt.primary_metric) else -attempt.cv_mean
    return metric, -attempt.latency_ms_per_1k, attempt.interp_norm


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _result(attempt: AttemptSummary, name: str) -> ConstraintResult | None:
    return next((r for r in attempt.constraint_results if r.name == name), None)


def _gap(result: ConstraintResult | None) -> float:
    """0 when the constraint passes; otherwise how far the measurement is from it."""
    if result is None:
        return float("inf")
    if result.passed:
        return 0.0
    if not isinstance(result.measured, (int, float)) or not isinstance(result.threshold, (int, float)):
        return 1.0
    scale = abs(float(result.threshold)) or 1.0
    return abs(float(result.measured) - float(result.threshold)) / scale
