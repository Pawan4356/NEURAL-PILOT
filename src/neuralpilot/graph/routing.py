"""Routing rules: which node runs next. Pure functions of graph state.

Every router checks for a recorded node error first, so any failure goes
to the `fail` node, which persists the run. The routers never call an LLM
and never touch data; they only read fields the nodes have written.

    understand ─┬─> clarify ─┬─> prepare -> retrieve_knowledge -> start_iteration
                │            └─> clarify (invalid answer, re-ask)
                └─> prepare
    start_iteration ─┬─> plan -> resolve_plan ─┬─> execute -> reflect -> decide
                     └─> finalize | fail       ├─> plan (retry)
                                               └─> finalize | fail
    decide ─┬─> start_iteration (refine / replan)
            └─> finalize -> persist -> END
    fail -> END
"""

from __future__ import annotations

from typing import Callable, Literal

from .. import config

FAIL = "fail"


def _failed(state: dict) -> bool:
    return bool(state.get("error"))


def after(next_node: str) -> Callable[[dict], str]:
    """Router for a fixed edge that still diverts to `fail` on error."""

    def route(state: dict) -> str:
        return FAIL if _failed(state) else next_node

    route.__name__ = f"route_to_{next_node}"
    return route


def after_understand(state: dict) -> Literal["clarify", "prepare", "fail"]:
    if _failed(state):
        return FAIL
    if state.get("clarification_question"):
        return "clarify"
    return "prepare"


def after_clarify(state: dict) -> Literal["clarify", "prepare", "fail"]:
    """An answer that is not a column is asked again, a bounded number of times."""
    if _failed(state):
        return FAIL
    if state.get("clarification_question"):
        if state.get("clarification_rounds", 0) >= config.MAX_CLARIFICATION_ROUNDS:
            return FAIL
        return "clarify"
    return "prepare"


def _stop_with_best(state: dict) -> Literal["finalize", "fail"]:
    """A forced stop keeps the best attempt if there is one."""
    return "finalize" if state.get("best") is not None else FAIL


def after_start_iteration(state: dict) -> Literal["plan", "finalize", "fail"]:
    if _failed(state):
        return FAIL
    decision = state.get("decision")
    if decision is not None and decision.forced and decision.decision == "stop":
        return _stop_with_best(state)
    return "plan"


def after_resolve_plan(state: dict) -> Literal["execute", "plan", "finalize", "fail"]:
    if _failed(state):
        return FAIL
    if state.get("plan") is not None:
        return "execute"
    decision = state.get("decision")
    if decision is not None and decision.forced and decision.decision == "stop":
        return _stop_with_best(state)  # every registry configuration was tried
    if state.get("plan_attempt", 0) >= config.MAX_PLAN_ATTEMPTS:
        return FAIL  # the node records why; this is a safety net
    return "plan"


def after_decide(state: dict) -> Literal["start_iteration", "finalize", "fail"]:
    if _failed(state):
        return FAIL
    decision = state["decision"]
    if decision.decision in ("accept", "stop"):
        return "finalize"
    return "start_iteration"


def recursion_limit(max_iterations: int) -> int:
    """Upper bound on graph steps for one run, so a routing bug cannot loop
    forever. Per iteration: start, (plan, resolve) x MAX_PLAN_ATTEMPTS,
    execute, reflect, decide. Plus understanding/clarification up front and
    finalize/persist (or fail) at the end."""
    per_iteration = 4 + 2 * config.MAX_PLAN_ATTEMPTS
    setup = 4 + config.MAX_CLARIFICATION_ROUNDS
    return setup + per_iteration * max(max_iterations, 1) + 4
