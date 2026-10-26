"""Phase 9: deterministic routing rules (Deliverable 6.4), table-driven."""

import pytest

from neuralpilot.graph.decision import decide_route


def route(**overrides):
    args = dict(
        cancelled=False,
        budget_exhausted=False,
        iteration=2,
        max_iterations=5,
        best_feasible=True,
        score_history=[0.6, 0.9],
        weakest_block_history=["model_family", "hyperparameters"],
        proposed_action="none",
    )
    return decide_route(**{**args, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"cancelled": True},
        {"budget_exhausted": True},
        {"iteration": 5, "max_iterations": 5},
    ],
)
def test_guardrails_force_a_stop(overrides):
    decision = route(**overrides)
    assert decision.decision == "stop"
    assert decision.forced


def test_a_feasible_best_is_accepted_on_the_second_iteration_when_nothing_is_proposed():
    decision = route()
    assert decision.decision == "accept"
    assert not decision.forced


@pytest.mark.parametrize("proposed", ["none", "change_model", "address_imbalance"])
def test_accept_is_never_granted_on_the_first_iteration(proposed):
    decision = route(iteration=1, score_history=[0.9], weakest_block_history=["model_family"], proposed_action=proposed)
    assert decision.decision != "accept"


def test_an_infeasible_best_is_never_accepted():
    decision = route(best_feasible=False)
    assert decision.decision == "replan"


def test_a_change_with_a_small_last_gain_is_accepted():
    decision = route(score_history=[0.900, 0.901], proposed_action="change_model")
    assert decision.decision == "accept"


def test_a_change_with_a_large_last_gain_is_refined():
    decision = route(score_history=[0.5, 0.9], proposed_action="change_preprocessing")
    assert decision.decision == "refine"
    assert not decision.forced


def test_no_proposed_change_with_a_large_gain_replans():
    decision = route(score_history=[0.5, 0.9], proposed_action="none", best_feasible=False)
    assert decision.decision == "replan"


def test_plateau_stops_an_infeasible_run_instead_of_replanning_forever():
    decision = route(
        best_feasible=False,
        score_history=[0.40, 0.50, 0.505, 0.510],
        weakest_block_history=["model_family", "model_family"],
        proposed_action="change_model",
    )
    assert decision.decision == "stop"
    assert decision.forced
    assert "plateau" in decision.rationale


def test_a_feasible_plateau_is_accepted_rather_than_forced_to_stop():
    decision = route(score_history=[0.40, 0.50, 0.505, 0.510], proposed_action="change_model")
    assert decision.decision == "accept"


def test_every_decision_has_a_rationale():
    for overrides in ({}, {"best_feasible": False}, {"cancelled": True}, {"iteration": 1}):
        assert route(**overrides).rationale
