"""Structured JSON contracts shared between agents and the execution layer.

These are the exact shapes described in §3 (agent I/O) and §4 (execution
layer). Pydantic gives us validation for free, which doubles as the
"structured JSON I/O" requirement for the LLM agents (§8 item 7).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProblemType = Literal["classification", "regression"]
PrimaryMetric = Literal[
    "accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "roc_auc", "r2", "rmse", "mae"
]
SplitStrategy = Literal["stratified", "random", "group", "time"]
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
    # Optional; when absent or invalid for the task, the engine picks a
    # deterministic default (see engine.evaluation.resolve_primary_metric).
    primary_metric: PrimaryMetric | None = None
    # Only set when the user explicitly asks for a time-ordered or grouped
    # evaluation; validated against the dataset's columns before use.
    time_column: str | None = None
    group_column: str | None = None


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
    # Upper bound (config.HPO_MAX_TRIALS) is enforced by clamping in the
    # engine and recorded as a plan adjustment, not rejected here.
    n_trials: int = Field(25, ge=0)
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


class SplitSpec(BaseModel):
    """How a run's data is divided. The holdout is fixed for the whole run;
    CV folds are drawn from the training part only."""

    strategy: SplitStrategy
    test_size: float = 0.2
    n_folds: int = 3
    seed: int = 42
    group_column: str | None = None
    time_column: str | None = None
    hpo_max_rows: int | None = None


class MetricSummary(BaseModel):
    mean: float
    std: float
    folds: list[float]


class ResourceUsage(BaseModel):
    train_s: float
    hpo_s: float
    peak_rss_mb: float | None = None
    latency_ms_per_1k: float
    model_bytes: int | None = None


class AttemptResult(BaseModel):
    """Measured outcome of one plan, computed from the training split only."""

    model_family: str  # resolved registry name actually executed
    primary_metric: str
    cv: dict[str, MetricSummary]
    best_params: dict = Field(default_factory=dict)
    n_trials_completed: int = 0
    hpo_stopped_reason: str | None = None
    hpo_rows: int | None = None  # rows used for HPO/CV after subsampling
    class_weight: str | None = None
    split: SplitSpec
    resources: ResourceUsage

    @property
    def cv_score(self) -> float:
        return self.cv[self.primary_metric].mean


class ExecutionResult(BaseModel):
    metrics: dict[str, float]
    latency_seconds: float
    diagnostics: list[BlockDiagnostics]
    shap_summary: dict | None = None
    fallbacks: list[dict] = Field(default_factory=list)
    attempt: AttemptResult | None = None
    plan_adjustments: list[str] = Field(default_factory=list)


class AttemptSummary(BaseModel):
    """Compact record of an executed plan, fed back to the Planning Agent so
    it never re-proposes a configuration that was already measured."""

    signature: str
    iteration: int
    model_family: str
    preprocessing: list[str]
    feature_engineering: list[str]
    tuning: str
    primary_metric: str
    cv_mean: float
    cv_std: float
    composite_score: float


class ValidationReport(BaseModel):
    """Multi-objective validation output (§4.4), also the Reflection Agent's input."""

    composite_score: float
    accuracy_norm: float
    latency_norm: float
    interp_norm: float
    raw_metrics: dict[str, float]
    per_block_diagnostics: list[BlockDiagnostics]
    primary_metric: str | None = None
    latency_ms_per_1k: float | None = None


class ReflectionOutput(BaseModel):
    """Reflection Agent output (§3.3)."""

    weakest_block: WeakestBlock
    rationale: str


class DecisionOutput(BaseModel):
    """Decision Agent output (§3.4)."""

    decision: Decision
    rationale: str
    forced: bool = False


class IterationLog(BaseModel):
    """One completed loop iteration, as shown in the UI and persisted."""

    iteration: int
    plan_summary: dict
    composite_score: float
    accuracy_norm: float
    latency_norm: float
    interp_norm: float
    weakest_block: str
    decision: str
    decision_rationale: str
    fallbacks: list[dict] = Field(default_factory=list)
    verification_warnings: list[str] = Field(default_factory=list)
    signature: str = ""
    plan_source: str = "planner"  # "planner" or "deterministic_fallback"
    primary_metric: str = ""
    cv_mean: float = 0.0
    cv_std: float = 0.0
    latency_ms_per_1k: float = 0.0
    decision_forced: bool = False
    plan_adjustments: list[str] = Field(default_factory=list)


class BestAttempt(BaseModel):
    """The best attempt so far, selected on CV — never by an LLM.

    Pure data so it can live in checkpointed graph state; the fitted
    pipeline is kept in `graph.resources` and on disk, not here."""

    iteration: int
    plan: PlanSpec
    summary: AttemptSummary
    validation: ValidationReport
    attempt: AttemptResult
