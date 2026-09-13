"""Execution Engine (§4.2) — deterministic, no LLM.

Runs preprocessing -> feature engineering -> training -> hyperparameter
tuning exactly as verified, wrapping scikit-learn/XGBoost/CatBoost/
Optuna/SHAP as callable modules. Agents select and parametrize; this
module executes and logs every step (inputs, outputs, timing, errors).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from .meta_features import is_categorical_column
from .registry import resolve_component
from .schemas import BlockDiagnostics, ExecutionResult, PlanSpec, ProblemType, RequirementSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ExecutionOutcome:
    """run_pipeline's return value. `result` is the pure-data ExecutionResult
    (scoring/reflection only ever see this); `pipeline` and `label_encoder`
    are the live fitted objects, kept separate so ExecutionResult stays a
    serializable pydantic model. The orchestrator uses `pipeline` to persist
    the final model artifact (§6 step 10: "Final output: model, ...").
    """

    result: ExecutionResult
    pipeline: Pipeline
    label_encoder: LabelEncoder | None

# ---------------------------------------------------------------------------
# Feature engineering step registry.
#
# Not one of the §4.3 component categories (encoder/scaler/model_family/
# tuner), so it does not go through the fallback decision tree — an
# unrecognized feature-engineering step is simply skipped and logged as a
# warning diagnostic, never a run failure.
# ---------------------------------------------------------------------------
_FEATURE_ENGINEERING: dict[str, Callable[..., object]] = {
    "PolynomialFeatures": lambda **kw: PolynomialFeatures(degree=kw.get("degree", 2), include_bias=False),
    "SelectKBest": lambda **kw: SelectKBest(
        score_func=f_classif if kw.get("problem_type") == "classification" else f_regression,
        k=kw.get("k", 10),
    ),
    "PCA": lambda **kw: PCA(n_components=kw.get("n_components", 0.95)),
}

# ---------------------------------------------------------------------------
# Hyperparameter search spaces, keyed by model_family registry name.
# ---------------------------------------------------------------------------


def _search_space(model_family: str, trial: optuna.Trial, problem_type: ProblemType) -> dict:
    if model_family == "LinearModel":
        if problem_type == "classification":
            return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        return {"alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True)}
    if model_family == "DecisionTree":
        return {"max_depth": trial.suggest_int("max_depth", 2, 20)}
    if model_family == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
        }
    if model_family == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    if model_family == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    if model_family == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 200),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    return {}


def _prepare_xy(
    df: pd.DataFrame, requirement: RequirementSpec, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], LabelEncoder | None]:
    data = df.dropna(subset=[requirement.target_column]).copy()
    y_raw = data[requirement.target_column]
    X = data.drop(columns=[requirement.target_column])

    label_encoder = None
    if requirement.problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index)
    else:
        y = y_raw

    categorical_cols = [c for c in X.columns if is_categorical_column(X[c])]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    stratify = y if requirement.problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, categorical_cols, numeric_cols, label_encoder


def _build_preprocessor(
    plan: PlanSpec, categorical_cols: list[str], numeric_cols: list[str], categorical_cardinality: int | None
) -> tuple[ColumnTransformer, list[dict]]:
    fallbacks: list[dict] = []
    encoder_name = next((s.component for s in plan.preprocessing if s.category == "encoder"), "OneHotEncoder")
    scaler_name = next((s.component for s in plan.preprocessing if s.category == "scaler"), "StandardScaler")

    encoder_spec, encoder_fb = resolve_component(
        "encoder", encoder_name, categorical_cardinality=categorical_cardinality
    )
    if encoder_fb:
        fallbacks.append(encoder_fb.to_dict())

    scaler_spec, scaler_fb = resolve_component("scaler", scaler_name)
    if scaler_fb:
        fallbacks.append(scaler_fb.to_dict())

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_spec.build())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder_spec.build()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, fallbacks


def _build_feature_engineering_steps(
    plan: PlanSpec, problem_type: ProblemType
) -> tuple[list[tuple[str, object]], list[str]]:
    steps: list[tuple[str, object]] = []
    diagnostics_warnings: list[str] = []
    for i, fe_step in enumerate(plan.feature_engineering):
        factory = _FEATURE_ENGINEERING.get(fe_step.name)
        if factory is None:
            diagnostics_warnings.append(f"unknown feature_engineering step '{fe_step.name}' skipped")
            continue
        params = dict(fe_step.params)
        params.setdefault("problem_type", problem_type)
        steps.append((f"fe_{i}_{fe_step.name}", factory(**params)))
    return steps, diagnostics_warnings


def _tune_hyperparameters(
    preprocessor: ColumnTransformer,
    fe_steps: list[tuple[str, object]],
    model_family: str,
    plan: PlanSpec,
    problem_type: ProblemType,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict, list[dict], BlockDiagnostics]:
    start = time.perf_counter()
    fallbacks: list[dict] = []

    tuner_spec, tuner_fb = resolve_component("tuner", plan.tuning.component)
    if tuner_fb:
        fallbacks.append(tuner_fb.to_dict())
    tuner_config = tuner_spec.build(n_trials=plan.tuning.n_trials)

    best_params: dict = {}
    n_trials = tuner_config["n_trials"]

    if tuner_config["sampler"] != "none" and n_trials > 0:
        model_spec, model_fb = resolve_component("model_family", model_family)
        if model_fb:
            fallbacks.append(model_fb.to_dict())

        sampler = (
            optuna.samplers.RandomSampler(seed=42)
            if tuner_config["sampler"] == "random"
            else optuna.samplers.TPESampler(seed=42)
        )
        scoring = "accuracy" if problem_type == "classification" else "r2"

        def objective(trial: optuna.Trial) -> float:
            params = _search_space(model_family, trial, problem_type)
            estimator = model_spec.build(problem_type, **params)
            pipeline = Pipeline([("preprocess", preprocessor), *fe_steps, ("model", estimator)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scoring)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=sampler)

        def report_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            message = f"Hyperparameter tuning: trial {len(study.trials)}/{n_trials}"
            if progress_callback:
                progress_callback(message)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[report_progress],
        )
        best_params = study.best_params

    duration = time.perf_counter() - start
    diagnostics = BlockDiagnostics(
        block="hyperparameters",
        duration_seconds=duration,
        details={"tuner": tuner_spec.name, "n_trials": n_trials, "best_params": best_params},
    )
    return best_params, fallbacks, diagnostics


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, problem_type: ProblemType) -> dict[str, float]:
    if problem_type == "classification":
        from sklearn.metrics import accuracy_score

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def _generate_shap_summary(
    pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, problem_type: ProblemType
) -> tuple[dict | None, BlockDiagnostics]:
    start = time.perf_counter()
    try:
        preprocess = Pipeline(pipeline.steps[:-1])
        model = pipeline.steps[-1][1]

        background = preprocess.transform(X_train.sample(min(50, len(X_train)), random_state=42))
        sample = preprocess.transform(X_test.sample(min(50, len(X_test)), random_state=42))
        try:
            feature_names = preprocess.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(sample.shape[1])]

        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(sample)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        if mean_abs.ndim > 1:  # multi-class: average across classes
            mean_abs = mean_abs.mean(axis=-1)

        importance = sorted(
            zip(feature_names, mean_abs.tolist()), key=lambda pair: pair[1], reverse=True
        )[:10]
        summary = {"top_features": [{"feature": f, "mean_abs_shap": v} for f, v in importance]}
        diagnostics = BlockDiagnostics(
            block="shap", duration_seconds=time.perf_counter() - start, details={"n_features": len(feature_names)}
        )
        return summary, diagnostics
    except Exception as exc:  # SHAP failures never fail the run
        diagnostics = BlockDiagnostics(
            block="shap", duration_seconds=time.perf_counter() - start, details={}, error=str(exc)
        )
        return None, diagnostics


def run_pipeline(
    plan: PlanSpec,
    requirement: RequirementSpec,
    df: pd.DataFrame,
    *,
    generate_shap: bool = False,
    random_state: int = 42,
    progress_callback: Callable[[str], None] | None = None,
) -> ExecutionOutcome:
    diagnostics: list[BlockDiagnostics] = []
    fallbacks: list[dict] = []

    # --- split ---
    t0 = time.perf_counter()
    X_train, X_test, y_train, y_test, categorical_cols, numeric_cols, label_encoder = _prepare_xy(
        df, requirement, random_state
    )
    categorical_cardinality = (
        max((X_train[c].nunique(dropna=True) for c in categorical_cols), default=0) if categorical_cols else None
    )
    diagnostics.append(
        BlockDiagnostics(
            block="split",
            duration_seconds=time.perf_counter() - t0,
            details={"n_train": len(X_train), "n_test": len(X_test)},
        )
    )

    # --- preprocessing ---
    t0 = time.perf_counter()
    preprocessor, preprocess_fallbacks = _build_preprocessor(
        plan, categorical_cols, numeric_cols, categorical_cardinality
    )
    fallbacks.extend(preprocess_fallbacks)
    diagnostics.append(
        BlockDiagnostics(
            block="preprocessing",
            duration_seconds=time.perf_counter() - t0,
            details={"categorical_cols": categorical_cols, "numeric_cols": numeric_cols},
        )
    )

    # --- feature engineering ---
    t0 = time.perf_counter()
    fe_steps, fe_warnings = _build_feature_engineering_steps(plan, requirement.problem_type)
    diagnostics.append(
        BlockDiagnostics(
            block="feature_engineering",
            duration_seconds=time.perf_counter() - t0,
            details={"steps": [name for name, _ in fe_steps], "warnings": fe_warnings},
        )
    )

    # --- hyperparameter tuning ---
    best_params, tuning_fallbacks, tuning_diag = _tune_hyperparameters(
        preprocessor,
        fe_steps,
        plan.model_family,
        plan,
        requirement.problem_type,
        X_train,
        y_train,
        progress_callback=progress_callback,
    )
    fallbacks.extend(tuning_fallbacks)
    diagnostics.append(tuning_diag)

    # --- final training ---
    t0 = time.perf_counter()
    model_spec, model_fb = resolve_component("model_family", plan.model_family)
    if model_fb and model_fb.to_dict() not in fallbacks:
        fallbacks.append(model_fb.to_dict())
    estimator = model_spec.build(requirement.problem_type, **best_params)
    pipeline = Pipeline([("preprocess", preprocessor), *fe_steps, ("model", estimator)])
    if progress_callback:
        progress_callback(f"Final training of {model_spec.name}")
    pipeline.fit(X_train, y_train)
    diagnostics.append(
        BlockDiagnostics(
            block="model_family",
            duration_seconds=time.perf_counter() - t0,
            details={"model_family": model_spec.name, "params": best_params},
        )
    )

    # --- evaluation / latency ---
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    latency_seconds = time.perf_counter() - t0
    raw_metrics = _compute_metrics(y_test, y_pred, requirement.problem_type)
    diagnostics.append(
        BlockDiagnostics(block="evaluation", duration_seconds=latency_seconds, details=raw_metrics)
    )

    # --- SHAP (only if explicitly requested; never gated by score itself) ---
    shap_summary = None
    if generate_shap:
        if progress_callback:
            progress_callback("Generating SHAP feature importance")
        shap_summary, shap_diag = _generate_shap_summary(pipeline, X_train, X_test, requirement.problem_type)
        diagnostics.append(shap_diag)

    result = ExecutionResult(
        metrics=raw_metrics,
        latency_seconds=latency_seconds,
        diagnostics=diagnostics,
        shap_summary=shap_summary,
        fallbacks=fallbacks,
    )
    return ExecutionOutcome(result=result, pipeline=pipeline, label_encoder=label_encoder)
