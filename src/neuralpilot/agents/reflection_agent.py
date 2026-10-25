"""Reflection Agent (§3.3, Phase 9).

Input: the validation report of the planner's plan (scores, constraint
checks, per-block diagnostics), the attempts executed so far, and the budget
left. Output: the weakest block, what is wrong, and at most one kind of change
the planner should make.

The agent proposes; it does not decide. A violated constraint it names must
be one the report shows as failing: a claim that does not match is sent back
once, and anything still unmatched is dropped and noted in `guard_notes`.
Suggested components are checked later, against the registry and the
executed attempts, by `reflection_guard`.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..llm_client import LLMClient, get_default_client
from ..observability import traced
from ..reflection_guard import failing_constraints, split_violated_claims
from ..schemas import AttemptSummary, ReflectionOutput, UnsupportedRecommendation, ValidationReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Reflection Agent in the Neural Pilot system.
You never write or execute code. You read a validation report for one pipeline (composite \
score, its accuracy/latency/interpretability components, the constraint checks, and \
per-block execution diagnostics) and the attempts already executed in this run. Identify \
the single weakest block, say what is wrong, and propose at most one kind of change.

weakest_block must be exactly one of: "preprocessing", "feature_engineering", \
"model_family", "hyperparameters".
proposed_action must be exactly one of: "change_model", "change_preprocessing", \
"change_feature_engineering", "address_imbalance", "expand_search", "reduce_complexity", \
"none". Use "none" when you have no change you can justify from the report.
violated_constraints: only names of constraints that the report marks as failing. Never \
name a constraint that passed.
suggestions: concrete components from the registry, each {"category": encoder|scaler|\
model_family|tuner|feature_engineering, "component": string, "params": object, \
"rationale": string}. Do not repeat a configuration listed as already executed.

Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{"weakest_block": string, "rationale": string, "diagnosis": string, \
"violated_constraints": [string], "proposed_action": string, "suggestions": [object], \
"expected_effect": string, "confidence": number}
"""


def _user_prompt(
    report: ValidationReport,
    attempts: list[AttemptSummary],
    iteration: int,
    budget_left_s: float | None,
    unsupported: list[UnsupportedRecommendation],
) -> str:
    diag_lines = [
        f"- {d.block}: duration={d.duration_seconds:.3f}s, details={d.details}"
        + (f", error={d.error}" if d.error else "")
        for d in report.per_block_diagnostics
    ]
    failing = [
        f"- {r.name}: {r.explanation}" for r in report.constraint_results if not r.passed
    ]
    passing = [r.name for r in report.constraint_results if r.passed]
    history = [
        f"- {a.signature} (iteration {a.iteration}): {a.model_family}, CV {a.primary_metric}={a.cv_mean:.4f}, "
        f"feasible={a.feasible}"
        for a in attempts
    ]
    lines = [
        f"iteration={iteration}",
        f"budget_left_s={'unknown' if budget_left_s is None else round(budget_left_s, 1)}",
        f"composite_score={report.composite_score:.4f}",
        f"accuracy_norm={report.accuracy_norm:.4f}",
        f"latency_norm={report.latency_norm:.4f}",
        f"interp_norm={report.interp_norm:.4f}",
        f"raw_metrics={report.raw_metrics}",
        f"feasible={report.feasible}",
        "failing_constraints:" if failing else "failing_constraints: none",
        *failing,
        f"passing_constraints={passing}",
        "per_block_diagnostics:",
        *diag_lines,
        "attempts_executed_in_this_run:" if history else "attempts_executed_in_this_run: none",
        *history,
    ]
    if unsupported:
        lines.append("components_the_registry_could_not_use:")
        lines += [f"- {u.category} '{u.requested}' ({u.reason})" for u in unsupported]
    return "\n".join(lines) + "\n"


def _with_constraint_guard(
    first: ReflectionOutput,
    retry: Callable[[], ReflectionOutput],
    failing: list[str],
) -> ReflectionOutput:
    """Keep only the violated-constraint claims that match a failing constraint.
    A mismatch is sent back once; the retry is used whatever it says."""
    notes: list[str] = []
    output = first
    valid, invalid = split_violated_claims(output.violated_constraints, failing)
    if invalid:
        notes.append(f"constraint claims {invalid} did not match a failing constraint; asked again once")
        output = retry()
        valid, invalid = split_violated_claims(output.violated_constraints, failing)
    if invalid:
        notes.append(f"dropped constraint claims {invalid}: they do not match a failing constraint")
    for note in notes:
        logger.info("reflection guard: %s", note)
    return output.model_copy(update={"violated_constraints": valid, "guard_notes": [*output.guard_notes, *notes]})


@traced("reflection_agent", tags=["agent"])
def reflect(
    report: ValidationReport,
    client: LLMClient | None = None,
    *,
    attempts: list[AttemptSummary] | None = None,
    iteration: int = 1,
    budget_left_s: float | None = None,
    unsupported: list[UnsupportedRecommendation] | None = None,
) -> ReflectionOutput:
    client = client or get_default_client()
    prompt = _user_prompt(report, list(attempts or []), iteration, budget_left_s, list(unsupported or []))
    failing = failing_constraints(report.constraint_results)

    def ask_again() -> ReflectionOutput:
        note = (
            "\nYour previous answer named violated_constraints that do not match the failing constraints. "
            f"The failing constraints are: {failing or 'none'}. Answer again using only those names.\n"
        )
        return client.complete_json(SYSTEM_PROMPT, prompt + note, ReflectionOutput)

    first = client.complete_json(SYSTEM_PROMPT, prompt, ReflectionOutput)
    return _with_constraint_guard(first, ask_again, failing)
