"""Deterministic routing after reflection (Phase 9, Deliverable 6.4).

This replaces the LLM Decision Agent. Every input it used was numeric or a
claim the guards had already checked, so the rule is written out here and
tested as a table. The reflection is an input to the rule, not the verdict.

Order of the rules (the first that applies wins):
1. cancelled by the user -> stop (forced)
2. compute budget used up -> stop (forced)
3. iteration cap reached -> stop (forced)
4. the best attempt meets the hard constraints, the run has finished at
   least two iterations (so accept always rests on a measured change), and
   either nothing is proposed or the last change gained less than the plateau
   threshold -> accept
5. plateau on the score or the weakest block -> stop (forced)
6. the reflection proposed a change -> refine (the planner revises the block)
7. otherwise -> replan (the planner chooses a different strategy)

An infeasible best attempt is never accepted, so the run keeps searching until
a guardrail stops it. Repeated signatures are excluded by plan resolution.
"""

from __future__ import annotations

from .. import config
from ..plateau import detect_plateau, relative_improvement_pct
from ..schemas import DecisionOutput, ProposedAction


def decide_route(
    *,
    cancelled: bool,
    budget_exhausted: bool,
    iteration: int,
    max_iterations: int,
    best_feasible: bool,
    score_history: list[float],
    weakest_block_history: list[str],
    proposed_action: ProposedAction,
) -> DecisionOutput:
    if cancelled:
        return DecisionOutput(decision="stop", rationale="Run cancelled by user.", forced=True)
    if budget_exhausted:
        return DecisionOutput(
            decision="stop",
            rationale="Guardrail: the run's compute budget is exhausted; finalizing the best attempt so far.",
            forced=True,
        )
    if iteration >= max_iterations:
        return DecisionOutput(
            decision="stop",
            rationale=f"Guardrail: iteration_count ({iteration}) >= max_iterations ({max_iterations}).",
            forced=True,
        )

    if best_feasible and iteration >= 2 and _accepts(score_history, proposed_action):
        return DecisionOutput(decision="accept", rationale=_accept_reason(score_history, proposed_action))

    if detect_plateau(score_history, weakest_block_history):
        return DecisionOutput(
            decision="stop",
            rationale=f"Guardrail: plateau detected (score gain under {config.PLATEAU_SCORE_DELTA_PCT:g}% over "
            f"the last {config.PLATEAU_LOOKBACK_ITERS} iterations, or the same weakest block with no net gain).",
            forced=True,
        )

    if proposed_action != "none":
        return DecisionOutput(
            decision="refine",
            rationale=f"The reflection proposed '{proposed_action}' and the change is untried; revising that block.",
        )
    return DecisionOutput(
        decision="replan",
        rationale="No untried change was proposed; the planner should choose a different strategy.",
    )


def _accepts(score_history: list[float], proposed_action: ProposedAction) -> bool:
    if proposed_action == "none":
        return True
    return _last_gain_pct(score_history) < config.PLATEAU_SCORE_DELTA_PCT


def _last_gain_pct(score_history: list[float]) -> float:
    """Relative change of the planner's score over the last iteration (0 when there is no history)."""
    if len(score_history) < 2:
        return 0.0
    return relative_improvement_pct(score_history[-2], score_history[-1])


def _accept_reason(score_history: list[float], proposed_action: ProposedAction) -> str:
    if proposed_action == "none":
        return "The best attempt meets the hard constraints and the reflection proposes no further change."
    gain = _last_gain_pct(score_history)
    return (
        f"The best attempt meets the hard constraints and the last iteration gained {gain:+.2f}% "
        f"(under {config.PLATEAU_SCORE_DELTA_PCT:g}%), so further search is not expected to help."
    )
