"""Structured JSON contracts shared between agents and the execution layer.

These are the exact shapes described in §3 (agent I/O) and §4 (execution
layer). Pydantic gives us validation for free, which doubles as the
"structured JSON I/O" requirement for the LLM agents (§8 item 7).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

ProblemType = Literal["classification", "regression"]
PrimaryMetric = Literal[
    "accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "roc_auc", "r2", "rmse", "mae"
]
SplitStrategy = Literal["stratified", "random", "group", "time"]
WeakestBlock = Literal["preprocessing", "feature_engineering", "model_family", "hyperparameters"]
Decision = Literal["accept", "refine", "replan", "stop"]
ConstraintName = Literal[
    "min_primary_metric",
    "max_latency_ms_per_1k",
    "max_train_seconds",
    "max_model_mb",
    "interpretability",
    "max_cv_std",
]
CandidateSource = Literal["planner", "baseline", "profile", "memory", "reflection"]


class Constraint(BaseModel):
    """A user requirement that a measured attempt must meet (Deliverable 9.2).

    Hard constraints decide feasibility; soft ones are reported only."""

    name: ConstraintName
    kind: Literal["hard", "soft"] = "hard"
    # Numbers for the measured limits; "glass_box" | "explainable" for interpretability.
    threshold: float | str
    source: Literal["user_form", "user_text", "default"] = "user_text"

    @model_validator(mode="after")
    def _threshold_matches_name(self) -> "Constraint":
        if self.name == "interpretability":
            if self.threshold not in ("glass_box", "explainable"):
                raise ValueError("interpretability threshold must be 'glass_box' or 'explainable'")
        elif isinstance(self.threshold, str):
            raise ValueError(f"{self.name} needs a numeric threshold")
        return self


class ConstraintResult(BaseModel):
    name: str
    kind: Literal["hard", "soft"]
    threshold: float | str
    measured: float | str | None
    passed: bool
    explanation: str


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
    # Hard and soft requirements stated by the user (Phase 5). Entries the
    # validator does not understand are dropped and noted in `assumptions`.
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_constraints(cls, data):
        if not isinstance(data, dict) or not data.get("constraints"):
            return data
        kept: list[Constraint] = []
        dropped: list = []
        for item in data["constraints"]:
            try:
                kept.append(item if isinstance(item, Constraint) else Constraint.model_validate(item))
            except ValidationError:
                dropped.append(item)
        if dropped:
            data = {
                **data,
                "assumptions": [*(data.get("assumptions") or []), f"ignored {len(dropped)} unsupported constraint(s)"],
            }
        return {**data, "constraints": kept}


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
    # Ids of the documentation passages this plan relies on (Phase 8). Ids that
    # were not retrieved for this run are dropped before the plan executes.
    evidence_ids: list[str] = Field(default_factory=list)


class CandidatePlan(BaseModel):
    """One plan the engine will execute in an iteration. The planner's plan
    is always first; the others are deterministic challengers (Deliverable 8.3)."""

    source: CandidateSource
    plan: PlanSpec
    # Phase 7 warm start: hyperparameters a similar past run found for this
    # model family. Trial 1 of the search starts from them, if they still fit.
    warm_start_params: dict = Field(default_factory=dict)


UnsupportedReason = Literal[
    "not_in_registry",
    "incompatible_task",
    "incompatible_data",
    "invalid_params",
    "over_budget",
    "violates_constraint",
]


class UnsupportedRecommendation(BaseModel):
    """A component the planner (or a generator) asked for that the registry
    could not use as requested. Stored for the planner and for developers
    (which components are missing); never counted as an implemented method."""

    iteration: int
    requested: str
    category: str
    source: CandidateSource
    reason: UnsupportedReason
    alternative_used: str | None = None
    semantic_change: bool = False
    context: dict = Field(default_factory=dict)


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
    source: CandidateSource = "planner"
    latency_ms_per_1k: float = 0.0
    interp_norm: float = 0.5
    feasible: bool = True
    constraint_results: list[ConstraintResult] = Field(default_factory=list)
    best_params: dict = Field(default_factory=dict)  # stored for warm starts (Phase 7)


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
    # Phase 5: user constraints checked against this attempt's measured values.
    constraint_results: list[ConstraintResult] = Field(default_factory=list)
    feasible: bool = True


ProposedAction = Literal[
    "change_model",
    "change_preprocessing",
    "change_feature_engineering",
    "address_imbalance",
    "expand_search",
    "reduce_complexity",
    "none",
]


class ReflectionSuggestion(BaseModel):
    """One concrete change the Reflection Agent proposes. It is checked against
    the registry and the executed attempts before anything uses it."""

    category: Literal["encoder", "scaler", "model_family", "tuner", "feature_engineering"]
    component: str
    params: dict = Field(default_factory=dict)
    rationale: str = ""


class ReflectionOutput(BaseModel):
    """Reflection Agent output (§3.3, Phase 9 / Deliverable 9.5).

    The first two fields are the original contract. The rest are optional, so
    a reply that only names a weakest block still parses. `violated_constraints`
    must name constraints the validation report shows as failing; `guard_notes`
    records what the guard removed."""

    weakest_block: WeakestBlock
    rationale: str
    diagnosis: str = ""
    violated_constraints: list[str] = Field(default_factory=list)
    proposed_action: ProposedAction = "none"
    suggestions: list[ReflectionSuggestion] = Field(default_factory=list)
    expected_effect: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    guard_notes: list[str] = Field(default_factory=list)


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
    # One entry per candidate executed this iteration (Deliverable 8.5).
    candidates: list[dict] = Field(default_factory=list)


class BestAttempt(BaseModel):
    """The best attempt so far, selected on CV — never by an LLM.

    Pure data so it can live in checkpointed graph state; the fitted
    pipeline is kept in `graph.resources` and on disk, not here."""

    iteration: int
    plan: PlanSpec
    summary: AttemptSummary
    validation: ValidationReport
    attempt: AttemptResult
