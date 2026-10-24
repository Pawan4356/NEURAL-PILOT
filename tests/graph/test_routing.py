"""Phase 3: every routing rule, table-driven. Routers are pure functions of
state, so these need no graph, data or LLM."""

import pytest

from neuralpilot import config
from neuralpilot.graph import routing
from neuralpilot.schemas import DecisionOutput, PlanSpec, TuningStrategy

FORCED_STOP = DecisionOutput(decision="stop", rationale="guardrail", forced=True)
BEST = object()  # routers only check presence
PLAN = PlanSpec(preprocessing=[], model_family="LinearModel", tuning=TuningStrategy())


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"error": "boom"}, "fail"),
        ({"clarification_question": "which column?"}, "clarify"),
        ({"clarification_question": None}, "prepare"),
        ({}, "prepare"),
    ],
)
def test_after_understand(state, expected):
    assert routing.after_understand(state) == expected


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"error": "boom"}, "fail"),
        ({"clarification_question": None, "clarification_rounds": 1}, "prepare"),
        ({"clarification_question": "again?", "clarification_rounds": 1}, "clarify"),
        ({"clarification_question": "again?", "clarification_rounds": config.MAX_CLARIFICATION_ROUNDS}, "fail"),
    ],
)
def test_after_clarify(state, expected):
    assert routing.after_clarify(state) == expected


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"error": "boom"}, "fail"),
        ({}, "plan"),
        ({"decision": DecisionOutput(decision="refine", rationale="r")}, "plan"),
        ({"decision": FORCED_STOP, "best": BEST}, "finalize"),
        ({"decision": FORCED_STOP, "best": None}, "fail"),
    ],
)
def test_after_start_iteration(state, expected):
    assert routing.after_start_iteration(state) == expected


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"error": "verification failed 3 times"}, "fail"),
        ({"plan": PLAN, "plan_attempt": 1}, "execute"),
        ({"plan": None, "plan_attempt": 1}, "plan"),
        ({"plan": None, "plan_attempt": 2}, "plan"),
        ({"plan": None, "plan_attempt": config.MAX_PLAN_ATTEMPTS}, "fail"),
        ({"plan": None, "plan_attempt": 2, "decision": FORCED_STOP, "best": BEST}, "finalize"),
        ({"plan": None, "plan_attempt": 2, "decision": FORCED_STOP, "best": None}, "fail"),
    ],
)
def test_after_resolve_plan(state, expected):
    assert routing.after_resolve_plan(state) == expected


@pytest.mark.parametrize(
    "decision, expected",
    [("accept", "finalize"), ("stop", "finalize"), ("refine", "start_iteration"), ("replan", "start_iteration")],
)
def test_after_decide(decision, expected):
    assert routing.after_decide({"decision": DecisionOutput(decision=decision, rationale="r")}) == expected


def test_after_decide_error_wins():
    assert routing.after_decide({"error": "x", "decision": DecisionOutput(decision="accept", rationale="r")}) == "fail"


def test_fixed_edges_divert_to_fail_on_error():
    route = routing.after("reflect")
    assert route({}) == "reflect"
    assert route({"error": "boom"}) == "fail"


def test_recursion_limit_grows_with_iterations_and_covers_a_full_run():
    one, five = routing.recursion_limit(1), routing.recursion_limit(5)
    assert five > one
    # Worst case for one iteration: setup + all plan retries + execute/reflect/decide + finalize/persist.
    worst_case_one_iteration = 1 + config.MAX_CLARIFICATION_ROUNDS + 2 + 1 + 2 * config.MAX_PLAN_ATTEMPTS + 3 + 2
    assert one >= worst_case_one_iteration
