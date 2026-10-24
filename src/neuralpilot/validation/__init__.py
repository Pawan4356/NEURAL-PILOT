"""Constraint checks and validation reports (Phase 5).

Constraints come from the user (extracted by the requirement agent, checked
here). Measurements come from `engine.AttemptResult`; nothing in this
package calls an LLM.
"""

from .constraints import check_constraints, is_feasible, requires_glass_box, violated_names
from .report import evaluate_attempt, pareto_front, summarize_run

__all__ = [
    "check_constraints",
    "evaluate_attempt",
    "is_feasible",
    "pareto_front",
    "requires_glass_box",
    "summarize_run",
    "violated_names",
]
