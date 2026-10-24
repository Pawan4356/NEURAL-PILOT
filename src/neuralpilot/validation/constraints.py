"""Constraint checks on measured attempts (Phase 5, Deliverable 9.2-9.3).

Every check reads an `AttemptResult` (measured numbers) or the attempt's
interpretability class. The LLM only supplies the constraints themselves;
it never supplies the values being compared, and the explanation text is
built here from the numbers, so the report cannot say something the
measurements do not support.

Hard constraints decide feasibility. Soft ones are reported but do not
change which attempt wins.
"""

from __future__ import annotations

from typing import Iterable

from .. import config
from ..engine.evaluation import greater_is_better
from ..schemas import AttemptResult, Constraint, ConstraintResult, RequirementSpec


def requires_glass_box(constraints: Iterable[Constraint]) -> bool:
    """A hard "glass_box" requirement limits the search to linear and tree models."""
    return any(
        c.kind == "hard" and c.name == "interpretability" and c.threshold == "glass_box" for c in constraints
    )


def requirement_requires_glass_box(requirement: RequirementSpec) -> bool:
    return requires_glass_box(requirement.constraints)


def check_constraints(
    constraints: Iterable[Constraint], attempt: AttemptResult, *, interp_class: str | None
) -> list[ConstraintResult]:
    return [_check(c, attempt, interp_class) for c in constraints]


def is_feasible(results: Iterable[ConstraintResult]) -> bool:
    """True when every hard constraint passed."""
    return all(r.passed for r in results if r.kind == "hard")


def violated_names(results: Iterable[ConstraintResult]) -> list[str]:
    return [r.name for r in results if not r.passed]


def _check(constraint: Constraint, attempt: AttemptResult, interp_class: str | None) -> ConstraintResult:
    if constraint.name == "min_primary_metric":
        return _check_metric(constraint, attempt)
    if constraint.name == "interpretability":
        return _check_interpretability(constraint, interp_class)

    if constraint.name == "max_latency_ms_per_1k":
        measured = attempt.resources.latency_ms_per_1k
        return _at_most(constraint, measured, "latency", "ms/1k")
    if constraint.name == "max_train_seconds":
        return _at_most(constraint, attempt.resources.train_s, "training time", "s")
    if constraint.name == "max_model_mb":
        if attempt.resources.model_bytes is None:
            return _unavailable(constraint, "model size was not measured")
        return _at_most(constraint, attempt.resources.model_bytes / 1_000_000, "model size", "MB")
    if constraint.name == "max_cv_std":
        measured = attempt.cv[attempt.primary_metric].std
        return _at_most(constraint, measured, f"CV std of {attempt.primary_metric}", "")
    raise ValueError(f"unknown constraint {constraint.name}")


def _check_metric(constraint: Constraint, attempt: AttemptResult) -> ConstraintResult:
    metric = attempt.primary_metric
    measured = attempt.cv_score
    threshold = float(constraint.threshold)
    if greater_is_better(metric):
        passed = measured >= threshold
        if passed:
            text = f"CV {metric} {measured:.4f} meets the minimum {threshold:.4f}"
        else:
            text = f"CV {metric} {measured:.4f} is below the minimum {threshold:.4f} ({measured - threshold:+.4f})"
    else:
        passed = measured <= threshold
        if passed:
            text = f"CV {metric} {measured:.4f} is within the maximum {threshold:.4f}"
        else:
            text = f"CV {metric} {measured:.4f} exceeds the maximum {threshold:.4f} ({measured - threshold:+.4f})"
    return _result(constraint, measured, passed, text)


def _check_interpretability(constraint: Constraint, interp_class: str | None) -> ConstraintResult:
    measured = interp_class or "unknown"
    if constraint.threshold == "glass_box":
        passed = interp_class in config.GLASS_BOX_CLASSES
        text = (
            f"model class '{measured}' is glass box"
            if passed
            else f"model class '{measured}' is not glass box (needs one of {', '.join(config.GLASS_BOX_CLASSES)})"
        )
    else:
        # Every family has an explainer (permutation importance at least).
        passed = True
        text = f"model class '{measured}' has an explainer"
    return _result(constraint, measured, passed, text)


def _at_most(constraint: Constraint, measured: float, label: str, unit: str) -> ConstraintResult:
    threshold = float(constraint.threshold)
    suffix = f" {unit}" if unit else ""
    if measured <= threshold:
        text = f"{label} {measured:.1f}{suffix} is within the limit {threshold:.1f}{suffix}"
        return _result(constraint, measured, True, text)
    overshoot = (measured - threshold) / threshold * 100 if threshold else float("inf")
    text = f"{label} {measured:.1f}{suffix} > {threshold:.1f}{suffix} limit (+{overshoot:.0f}%)"
    return _result(constraint, measured, False, text)


def _unavailable(constraint: Constraint, reason: str) -> ConstraintResult:
    # An unmeasurable limit cannot be shown to pass, so it counts as failed.
    return _result(constraint, None, False, f"cannot be checked: {reason}")


def _result(constraint: Constraint, measured, passed: bool, explanation: str) -> ConstraintResult:
    return ConstraintResult(
        name=constraint.name,
        kind=constraint.kind,
        threshold=constraint.threshold,
        measured=measured,
        passed=passed,
        explanation=explanation,
    )
