"""Structured JSON contracts shared between agents and the execution layer.

These are the exact shapes described in §3 (agent I/O) and §4 (execution
layer). Pydantic gives us validation for free, which doubles as the
"structured JSON I/O" requirement for the LLM agents (§8 item 7).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProblemType = Literal["classification", "regression"]
WeakestBlock = Literal["preprocessing", "feature_engineering", "model_family", "hyperparameters"]
Decision = Literal["accept", "refine", "replan", "stop"]


class ConstraintWeights(BaseModel):
    accuracy: float = 0.7
    latency: float = 0.15
    interpretability: float = 0.15

    def normalized(self) -> "ConstraintWeights":
        total = self.accuracy + self.latency + self.interpretability
        if total <= 0:
            return ConstraintWeights()
        return ConstraintWeights(
            accuracy=self.accuracy / total,
            latency=self.latency / total,
            interpretability=self.interpretability / total,
        )


class RequirementSpec(BaseModel):
    """Requirement Understanding Agent output (§3.1)."""

    target_column: str
    target_confidence: float = Field(ge=0.0, le=1.0)
    problem_type: ProblemType
    constraint_weights: ConstraintWeights
    weight_confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)


class PreprocessingStep(BaseModel):
    category: Literal["encoder", "scaler"]
    component: str
    params: dict = Field(default_factory=dict)
    rationale: str = ""


class FeatureEngineeringStep(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)
    rationale: str = ""


class TuningStrategy(BaseModel):
    component: str = "OptunaRandom"
    n_trials: int = 25
    rationale: str = ""


class PlanSpec(BaseModel):
    """Planning Agent output (§3.2)."""

    preprocessing: list[PreprocessingStep]
    feature_engineering: list[FeatureEngineeringStep] = Field(default_factory=list)
    model_family: str
    tuning: TuningStrategy
    rationale: str = ""
    iteration: int = 1


class VerificationResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallbacks: list[dict] = Field(default_factory=list)


class BlockDiagnostics(BaseModel):
    block: str
    duration_seconds: float
    details: dict = Field(default_factory=dict)
    error: str | None = None


class ExecutionResult(BaseModel):
    metrics: dict[str, float]
    latency_seconds: float
    diagnostics: list[BlockDiagnostics]
    shap_summary: dict | None = None
    fallbacks: list[dict] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Multi-objective validation output (§4.4), also the Reflection Agent's input."""

    composite_score: float
    accuracy_norm: float
    latency_norm: float
    interp_norm: float
    raw_metrics: dict[str, float]
    per_block_diagnostics: list[BlockDiagnostics]


class ReflectionOutput(BaseModel):
    """Reflection Agent output (§3.3)."""

    weakest_block: WeakestBlock
    rationale: str


class DecisionOutput(BaseModel):
    """Decision Agent output (§3.4)."""

    decision: Decision
    rationale: str
    forced: bool = False
