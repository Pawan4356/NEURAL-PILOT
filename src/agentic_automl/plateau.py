"""Plateau Detection (§4.5) — deterministic, forces Decision Agent to `stop`.

Plateau = TRUE if either:
  (a) composite score improvement < 2% over the last 2 consecutive
      iterations (both of the last two iteration-over-iteration deltas), OR
  (b) the same weakest_block was identified in the last 2 consecutive
      Reflection Agent outputs with no net score gain across them.
"""

from __future__ import annotations

from . import config


def _relative_improvement_pct(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0 if current == 0 else 100.0
    return ((current - previous) / abs(previous)) * 100.0


def detect_plateau(score_history: list[float], weakest_block_history: list[str]) -> bool:
    n = config.PLATEAU_LOOKBACK_ITERS

    # Condition (a): needs at least n+1 scores to have n consecutive deltas.
    if len(score_history) >= n + 1:
        deltas = [
            _relative_improvement_pct(score_history[i - 1], score_history[i])
            for i in range(len(score_history) - n, len(score_history))
        ]
        if all(delta < config.PLATEAU_SCORE_DELTA_PCT for delta in deltas):
            return True

    # Condition (b): needs at least n weakest_block entries and n+1 scores
    # (or n scores, using the gain across the block-repeat window).
    if len(weakest_block_history) >= n:
        last_n_blocks = weakest_block_history[-n:]
        if len(set(last_n_blocks)) == 1 and len(score_history) >= n:
            net_gain = score_history[-1] - score_history[-n]
            if net_gain <= 0:
                return True

    return False
