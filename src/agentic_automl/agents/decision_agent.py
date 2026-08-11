"""Decision Agent (§3.4).

Input: current composite score, score history, weakest_block, iteration
count, plateau flag. Output: accept | refine | replan | stop.

Deterministic overrides (hardcoded guardrails, not just prompt
instructions — the LLM is never even called when these fire):
- iteration_count >= max_iterations -> force "stop"
- plateau condition met (§4.5) -> force "stop"
Otherwise the agent chooses freely among accept/refine/replan.
"""

from __future__ import annotations

from .. import config
from ..llm_client import LLMClient, get_default_client
from ..plateau import detect_plateau
from ..schemas import DecisionOutput

SYSTEM_PROMPT = """You are the Decision Agent in an agentic AutoML system.
You never write or execute code. You read the run's score history and the current \
weakest block, then choose exactly one of: "accept", "refine", "replan", "stop".

Guidance:
- "accept": the composite score is good and stable; finalize this run.
- "refine": the plan is fundamentally sound but the weakest_block should be revised \
  (small, targeted change, same overall strategy).
- "replan": the current strategy is not working; the Planning Agent should start over \
  with a different approach.
- "stop": give up further iteration (e.g. repeated failures with no path forward).
You will never be called when a plateau or max-iteration guardrail already forces "stop" — \
those are handled deterministically outside of you.

Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{"decision": "accept"|"refine"|"replan"|"stop", "rationale": string}
"""


def _user_prompt(
    composite_score: float,
    score_history: list[float],
    weakest_block: str,
    iteration_count: int,
    max_iterations: int,
) -> str:
    return (
        f"current_composite_score={composite_score:.4f}\n"
        f"score_history={score_history}\n"
        f"weakest_block={weakest_block}\n"
        f"iteration_count={iteration_count}\n"
        f"max_iterations={max_iterations}\n"
    )


def decide(
    *,
    composite_score: float,
    score_history: list[float],
    weakest_block: str,
    weakest_block_history: list[str],
    iteration_count: int,
    max_iterations: int = config.MAX_ITERATIONS,
    client: LLMClient | None = None,
) -> DecisionOutput:
    if iteration_count >= max_iterations:
        return DecisionOutput(
            decision="stop",
            rationale=f"Guardrail: iteration_count ({iteration_count}) >= max_iterations ({max_iterations}).",
            forced=True,
        )

    if detect_plateau(score_history, weakest_block_history):
        return DecisionOutput(
            decision="stop",
            rationale="Guardrail: plateau detected (score improvement < 2% over last 2 "
            "iterations, or same weakest_block repeated with no net gain).",
            forced=True,
        )

    client = client or get_default_client()
    output = client.complete_json(
        SYSTEM_PROMPT,
        _user_prompt(composite_score, score_history, weakest_block, iteration_count, max_iterations),
        DecisionOutput,
    )
    output.forced = False
    return output
