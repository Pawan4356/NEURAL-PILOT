"""Attempt runner — composes the engine components into one measured attempt.

    prepare_dataset  (once per run)  profile -> split -> leakage scan -> metric
    run_attempt      (per plan)      preprocess -> FE -> HPO (CV) -> CV summary -> fit -> latency
    holdout_evaluate (once per run)  score the chosen pipeline on the holdout

`run_attempt` only ever sees the training rows; the holdout rows are
touched by `holdout_evaluate` alone.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .. import config
from ..meta_features import is_categorical_column
from ..profiling import EXCLUDED_ROLES, ColumnProfile, LeakageFlag, leakage_scan, profile_columns
from ..registry import resolve_component
from ..schemas import (
    AttemptResult,
    BlockDiagnostics,
    ExecutionResult,
    PlanSpec,
    ProblemType,
    RequirementSpec,
    ResourceUsage,
)
from . import evaluation, hpo, splits
from .explainability import shap_summary
from .feature_engineering import build_fe_steps
from .model_factory import build_estimator, resolve_gpu, supports_class_weight
from .preprocessing import IMPUTATION, build_preprocessor, estimate_encoded_width
from .tracking import NoopTracker, Tracker
from .training import fit_final, peak_rss_mb


@dataclass
class PreparedData:
    """Everything about a run's data that must stay fixed across attempts."""

    target_column: str
    problem_type: ProblemType
    X: pd.DataFrame  # feature columns only, rows with a known target
    y: pd.Series
    label_encoder: LabelEncoder | None
    split: splits.DataSplit
    groups: pd.Series | None
    column_profiles: list[ColumnProfile]
    excluded_columns: dict[str, str]  # column -> reason
    leakage_flags: list[LeakageFlag]
    categorical_cols: list[str]
    numeric_cols: list[str]
    primary_metric: str
    balanced_class_weights: bool
    seed: int
    notes: list[str] = field(default_factory=list)

    @property
    def X_train(self) -> pd.DataFrame:
        return self.X.loc[self.split.train_index]

    @property
    def y_train(self) -> pd.Series:
        return self.y.loc[self.split.train_index]

    def categorical_cardinality(self) -> int | None:
        if not self.categorical_cols:
            return None
        X_train = self.X_train
        return int(max(X_train[c].nunique(dropna=True) for c in self.categorical_cols))

    def summary(self) -> dict:
        return {
            "n_rows": int(len(self.X)),
            "n_features_used": len(self.categorical_cols) + len(self.numeric_cols),
            "n_train": int(len(self.split.train_index)),
            "n_holdout": int(len(self.split.test_index)),
            "excluded_columns": dict(self.excluded_columns),
            "leakage_flags": [flag.to_dict() for flag in self.leakage_flags],
            "primary_metric": self.primary_metric,
            "balanced_class_weights": self.balanced_class_weights,
            "methodology": self.split.spec.model_dump(),
        }


@dataclass
class AttemptBudget:
    max_trials: int = config.HPO_MAX_TRIALS
    hpo_timeout_s: float | None = config.HPO_TIMEOUT_S


@dataclass
class AttemptOutcome:
    """`result` is pure data (scoring/reflection only see this); `pipeline`
    and `label_encoder` are the live fitted objects used for persistence."""

    result: ExecutionResult
    pipeline: Pipeline
    label_encoder: LabelEncoder | None
    signature: str = ""


def _needs_label_encoding(y: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return True
    # XGBoost and friends need classes 0..n-1.
    classes = sorted(y.unique())
    return classes != list(range(len(classes)))


def prepare_dataset(
    df: pd.DataFrame,
    requirement: RequirementSpec,
    *,
    seed: int = config.SEED,
    leakage_action: str = config.LEAKAGE_ACTION,
) -> PreparedData:
    target = requirement.target_column
    if target not in df.columns:
        raise ValueError(f"target_column '{target}' does not exist in dataset")
    data = df.dropna(subset=[target])
    notes: list[str] = []
    dropped = len(df) - len(data)
    if dropped:
        notes.append(f"{dropped} rows with a missing target were dropped")

    y_raw = data[target]
    label_encoder = None
    if requirement.problem_type == "classification" and _needs_label_encoding(y_raw):
        label_encoder = LabelEncoder()
        labels = y_raw if pd.api.types.is_numeric_dtype(y_raw) else y_raw.astype(str)
        y = pd.Series(label_encoder.fit_transform(labels), index=y_raw.index)
    else:
        y = y_raw

    profiles = profile_columns(data, target_column=target)
    excluded = {p.name: f"{p.role}: {p.reason}" for p in profiles if p.role in EXCLUDED_ROLES}

    spec = splits.build_split_spec(data, requirement, seed=seed, assumptions=notes)
    split = splits.make_holdout_split(data, y, spec)
    groups = splits.group_labels(data, spec)
    for col in (spec.group_column, spec.time_column):
        if col and col not in excluded:
            excluded[col] = "split column (used to define the evaluation split)"

    # Leakage scan on training rows only, over columns that would be features.
    candidates = [c for c in data.columns if c != target and c not in excluded]
    flags = leakage_scan(
        data.loc[split.train_index], target, requirement.problem_type, columns=candidates, seed=seed
    )
    for flag in flags:
        if leakage_action == "drop":
            excluded[flag.column] = f"suspected leakage: {flag.describe()}"
        else:
            notes.append(f"possible leakage (kept, LEAKAGE_ACTION=warn): {flag.describe()}")

    features = [c for c in data.columns if c != target and c not in excluded]
    if not features:
        raise ValueError(
            "No usable feature columns remain after excluding identifiers, constants, raw "
            f"timestamps, free text and suspected leakage: {excluded}"
        )
    X = data[features]
    categorical_cols = [c for c in features if is_categorical_column(X[c])]
    numeric_cols = [c for c in features if c not in categorical_cols]

    primary_metric, balanced, metric_notes = evaluation.resolve_primary_metric(
        requirement, y.loc[split.train_index]
    )
    notes.extend(metric_notes)

    return PreparedData(
        target_column=target,
        problem_type=requirement.problem_type,
        X=X,
        y=y,
        label_encoder=label_encoder,
        split=split,
        groups=groups,
        column_profiles=profiles,
        excluded_columns=excluded,
        leakage_flags=flags,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        primary_metric=primary_metric,
        balanced_class_weights=balanced,
        seed=seed,
        notes=notes,
    )


def plan_signature(plan: PlanSpec, *, categorical_cardinality: int | None = None) -> str:
    """Stable hash of what would actually execute: resolved component names,
    feature-engineering steps with params, imputation, and tuning budget."""
    encoder = next((s.component for s in plan.preprocessing if s.category == "encoder"), "OneHotEncoder")
    scaler = next((s.component for s in plan.preprocessing if s.category == "scaler"), "StandardScaler")
    encoder_spec, _ = resolve_component("encoder", encoder, categorical_cardinality=categorical_cardinality)
    scaler_spec, _ = resolve_component("scaler", scaler)
    model_spec, _ = resolve_component("model_family", plan.model_family)
    tuner_spec, _ = resolve_component("tuner", plan.tuning.component)
    n_trials = 0 if tuner_spec.name == "NoTuning" else min(plan.tuning.n_trials, config.HPO_MAX_TRIALS)
    payload = {
        "model_family": model_spec.name,
        "preprocessing": sorted([encoder_spec.name, scaler_spec.name]),
        "preprocessing_params": sorted(
            json.dumps({"c": s.component, "p": s.params}, sort_keys=True, default=str)
            for s in plan.preprocessing
            if s.params
        ),
        "feature_engineering": [
            json.dumps({"n": s.name, "p": s.params}, sort_keys=True, default=str) for s in plan.feature_engineering
        ],
        "imputation": IMPUTATION,
        "tuning": [tuner_spec.name, n_trials],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run_attempt(
    plan: PlanSpec,
    prepared: PreparedData,
    *,
    budget: AttemptBudget | None = None,
    generate_shap: bool = False,
    use_gpu: bool | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    tracker: Tracker | None = None,
) -> AttemptOutcome:
    budget = budget or AttemptBudget()
    tracker = tracker or NoopTracker()
    problem_type = prepared.problem_type
    diagnostics: list[BlockDiagnostics] = []
    fallbacks: list[dict] = []
    adjustments: list[str] = []

    gpu_enabled, gpu_fallback_reason = resolve_gpu(use_gpu)
    if gpu_fallback_reason:
        fallbacks.append({"category": "compute", "requested": "GPU", "used": "CPU", "reason": gpu_fallback_reason})

    # --- split (fixed for the run; this attempt only sees training rows) ---
    t0 = time.perf_counter()
    X_train, y_train = prepared.X_train, prepared.y_train
    spec = prepared.split.spec
    X_hpo, y_hpo = splits.subsample_for_hpo(X_train, y_train, spec, problem_type)
    groups_hpo = prepared.groups.loc[X_hpo.index].to_numpy() if prepared.groups is not None else None
    cv = splits.cv_splitter(spec, problem_type, y_hpo)
    categorical_cardinality = prepared.categorical_cardinality()
    signature = plan_signature(plan, categorical_cardinality=categorical_cardinality)
    diagnostics.append(
        BlockDiagnostics(
            block="split",
            duration_seconds=time.perf_counter() - t0,
            details={
                "strategy": spec.strategy,
                "n_train": len(X_train),
                "n_holdout": len(prepared.split.test_index),
                "n_hpo_rows": len(X_hpo),
                "cv_folds": cv.get_n_splits(),
            },
        )
    )

    # --- preprocessing ---
    t0 = time.perf_counter()
    preprocessor, preprocess_fallbacks, preprocess_adjustments = build_preprocessor(
        plan, prepared.categorical_cols, prepared.numeric_cols, categorical_cardinality
    )
    fallbacks.extend(preprocess_fallbacks)
    adjustments.extend(preprocess_adjustments)
    diagnostics.append(
        BlockDiagnostics(
            block="preprocessing",
            duration_seconds=time.perf_counter() - t0,
            details={
                "categorical_cols": prepared.categorical_cols,
                "numeric_cols": prepared.numeric_cols,
                "excluded_columns": prepared.excluded_columns,
            },
        )
    )

    # --- feature engineering (bounded) ---
    t0 = time.perf_counter()
    requested_encoder = next((s.component for s in plan.preprocessing if s.category == "encoder"), "OneHotEncoder")
    encoder_name = resolve_component(
        "encoder", requested_encoder, categorical_cardinality=categorical_cardinality
    )[0].name
    width = estimate_encoded_width(
        encoder_name,
        prepared.numeric_cols,
        [int(X_train[c].nunique(dropna=True)) for c in prepared.categorical_cols],
    )
    fe_steps, fe_warnings, fe_adjustments = build_fe_steps(plan, problem_type, width)
    adjustments.extend(fe_adjustments)
    diagnostics.append(
        BlockDiagnostics(
            block="feature_engineering",
            duration_seconds=time.perf_counter() - t0,
            details={"steps": [name for name, _ in fe_steps], "warnings": fe_warnings, "encoded_width": width},
        )
    )

    # --- model + tuner resolution (once; the search space uses the resolved name) ---
    model_spec, model_fb = resolve_component("model_family", plan.model_family)
    if model_fb:
        fallbacks.append(model_fb.to_dict())
    tuner_spec, tuner_fb = resolve_component("tuner", plan.tuning.component)
    if tuner_fb:
        fallbacks.append(tuner_fb.to_dict())
    requested_trials = plan.tuning.n_trials
    if requested_trials > budget.max_trials:
        adjustments.append(f"n_trials {requested_trials} clamped to {budget.max_trials}")
    tuner_config = tuner_spec.build(n_trials=min(requested_trials, budget.max_trials))
    balanced = prepared.balanced_class_weights and supports_class_weight(model_spec.name)
    if prepared.balanced_class_weights and not balanced:
        adjustments.append(f"{model_spec.name} has no native class weighting; trained unweighted")

    def make_pipeline(params: dict) -> Pipeline:
        estimator = build_estimator(
            model_spec, problem_type, params, gpu_enabled=gpu_enabled, balanced=balanced
        )
        return Pipeline([("preprocess", clone(preprocessor)), *[(n, clone(s)) for n, s in fe_steps], ("model", estimator)])

    tracker.start_attempt(signature, {"model_family": model_spec.name, "plan": plan.model_dump()})

    # --- hyperparameter tuning (CV on training rows, primary metric) ---
    n_classes = int(y_train.nunique()) if problem_type == "classification" else 0
    hpo_result = hpo.tune(
        make_pipeline,
        model_family=model_spec.name,
        problem_type=problem_type,
        sampler_name=tuner_config["sampler"],
        X=X_hpo,
        y=y_hpo,
        cv=cv,
        groups=groups_hpo,
        scorer_name=evaluation.sklearn_scorer(prepared.primary_metric, n_classes),
        n_trials=tuner_config["n_trials"],
        timeout_s=budget.hpo_timeout_s,
        seed=prepared.seed,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    diagnostics.append(
        BlockDiagnostics(
            block="hyperparameters",
            duration_seconds=hpo_result.duration_seconds,
            details={
                "tuner": tuner_spec.name,
                "n_trials": tuner_config["n_trials"],
                "n_trials_completed": hpo_result.n_trials_completed,
                "n_trials_pruned": hpo_result.n_trials_pruned,
                "n_trials_failed": hpo_result.n_trials_failed,
                "stopped_reason": hpo_result.stopped_reason,
                "objective": prepared.primary_metric,
                "best_params": hpo_result.best_params,
            },
        )
    )

    # --- CV summary of the chosen params (all metrics) ---
    t0 = time.perf_counter()
    best_params = dict(hpo_result.best_params)
    if progress_callback:
        progress_callback(f"Cross-validating {model_spec.name}")
    cv_summary = evaluation.cv_evaluate(
        make_pipeline(best_params), X_hpo, y_hpo, cv=cv, groups=groups_hpo, problem_type=problem_type
    )
    if prepared.primary_metric not in cv_summary:
        raise RuntimeError(
            f"Cross-validation could not compute the primary metric '{prepared.primary_metric}' "
            f"for {model_spec.name}"
        )
    cv_seconds = time.perf_counter() - t0

    # --- final training on the full training split ---
    pipeline = make_pipeline(best_params)
    if progress_callback:
        progress_callback(f"Final training of {model_spec.name}")
    train_seconds = fit_final(pipeline, X_train, y_train)
    diagnostics.append(
        BlockDiagnostics(
            block="model_family",
            duration_seconds=train_seconds,
            details={
                "model_family": model_spec.name,
                "params": best_params,
                "class_weight": "balanced" if balanced else None,
                "device": "cuda" if gpu_enabled and model_spec.name in {"XGBoost", "CatBoost"} else "cpu",
            },
        )
    )

    # --- latency / resources ---
    latency_sample = X_train.sample(min(config.LATENCY_SAMPLE_ROWS, len(X_train)), random_state=prepared.seed)
    latency_ms_per_1k = evaluation.measure_latency_ms_per_1k(pipeline, latency_sample)
    resources = ResourceUsage(
        train_s=train_seconds,
        hpo_s=hpo_result.duration_seconds,
        peak_rss_mb=peak_rss_mb(),
        latency_ms_per_1k=latency_ms_per_1k,
        model_bytes=evaluation.model_size_bytes(pipeline),
    )
    cv_metrics = evaluation.flat_metrics(cv_summary)
    diagnostics.append(
        BlockDiagnostics(
            block="evaluation",
            duration_seconds=cv_seconds,
            details={
                "method": f"{cv.get_n_splits()}-fold CV on training rows",
                "primary_metric": prepared.primary_metric,
                "cv": {k: {"mean": v.mean, "std": v.std} for k, v in cv_summary.items()},
                "latency_ms_per_1k": latency_ms_per_1k,
            },
        )
    )

    shap_result = None
    if generate_shap:
        if progress_callback:
            progress_callback("Generating SHAP feature importance")
        shap_result, shap_diag = shap_summary(pipeline, X_train, X_train, seed=prepared.seed)
        diagnostics.append(shap_diag)

    attempt = AttemptResult(
        model_family=model_spec.name,
        primary_metric=prepared.primary_metric,
        cv=cv_summary,
        best_params=best_params,
        n_trials_completed=hpo_result.n_trials_completed,
        hpo_stopped_reason=hpo_result.stopped_reason,
        hpo_rows=len(X_hpo),
        class_weight="balanced" if balanced else None,
        split=spec,
        resources=resources,
    )
    tracker.log_metrics(cv_metrics)
    tracker.end_attempt("ok")

    result = ExecutionResult(
        metrics=cv_metrics,
        latency_seconds=latency_ms_per_1k / 1000.0,
        diagnostics=diagnostics,
        shap_summary=shap_result,
        fallbacks=fallbacks,
        attempt=attempt,
        plan_adjustments=adjustments,
    )
    return AttemptOutcome(result=result, pipeline=pipeline, label_encoder=prepared.label_encoder, signature=signature)


def holdout_evaluate(pipeline: Pipeline, prepared: PreparedData) -> dict[str, float]:
    """Score a fitted pipeline on the run's holdout. Call once per run."""
    test_index = prepared.split.test_index
    return evaluation.compute_metrics(
        pipeline, prepared.X.loc[test_index], prepared.y.loc[test_index], prepared.problem_type
    )
