"""Multi-Objective Validation (§4.4) — deterministic weighted-sum scoring.

    score = w_acc * accuracy_norm + w_lat * (1 - latency_norm) + w_interp * interp_norm

No LLM involvement. Weights come from the Requirement Understanding Agent
output; everything else is computed from execution results.
"""

from __future__ import annotations

from . import config
from .registry import get_interp_class
from .schemas import BlockDiagnostics, ConstraintWeights, ProblemType, ValidationReport

DEFAULT_INTERP_NORM = 0.5  # unknown model family — neither rewarded nor punished


def _accuracy_norm(raw_metrics: dict[str, float], problem_type: ProblemType) -> float:
    if problem_type == "classification":
        value = raw_metrics.get("f1", raw_metrics.get("accuracy"))
        if value is None:
            raise ValueError("raw_metrics must contain 'f1' or 'accuracy' for classification")
        return max(0.0, min(1.0, float(value)))

    # regression: normalized R², clipped into [0, 1] since R² can be negative
    r2 = raw_metrics.get("r2")
    if r2 is not None:
        return max(0.0, min(1.0, float(r2)))

    # fall back to normalized RMSE if R² wasn't supplied: 1 / (1 + RMSE)
    rmse = raw_metrics.get("rmse")
    if rmse is None:
        raise ValueError("raw_metrics must contain 'r2' or 'rmse' for regression")
    return 1.0 / (1.0 + max(0.0, float(rmse)))


def _latency_norm(latency_seconds: float, rolling_max_latency: float) -> float:
    if rolling_max_latency <= 0:
        return 0.0
    return max(0.0, min(1.0, latency_seconds / rolling_max_latency))


def _interp_norm(model_family: str) -> float:
    interp_class = get_interp_class(model_family)
    if interp_class is None:
        return DEFAULT_INTERP_NORM
    return config.INTERPRETABILITY_TABLE.get(interp_class, DEFAULT_INTERP_NORM)


def compute_validation(
    *,
    raw_metrics: dict[str, float],
    latency_seconds: float,
    rolling_max_latency: float,
    model_family: str,
    problem_type: ProblemType,
    weights: ConstraintWeights,
    per_block_diagnostics: list[BlockDiagnostics],
) -> ValidationReport:
    w = weights.normalized()

    accuracy_norm = _accuracy_norm(raw_metrics, problem_type)
    latency_norm = _latency_norm(latency_seconds, rolling_max_latency)
    interp_norm = _interp_norm(model_family)

    composite_score = (
        w.accuracy * accuracy_norm + w.latency * (1 - latency_norm) + w.interpretability * interp_norm
    )

    return ValidationReport(
        composite_score=composite_score,
        accuracy_norm=accuracy_norm,
        latency_norm=latency_norm,
        interp_norm=interp_norm,
        raw_metrics=raw_metrics,
        per_block_diagnostics=per_block_diagnostics,
    )


def should_generate_shap(weights: ConstraintWeights, weight_confidence: float) -> bool:
    """SHAP generation is gated on interpretability being explicitly requested,
    not on the composite score (§4.4): a high, confidently-stated
    interpretability weight means the user asked for it explicitly.
    """
    return weight_confidence >= config.WEIGHT_CONFIDENCE_THRESHOLD and weights.interpretability > (
        config.DEFAULT_CONSTRAINT_WEIGHTS["interpretability"] + 1e-9
    )
