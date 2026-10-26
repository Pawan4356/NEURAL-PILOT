"""Evaluation (engine component 7).

One primary metric drives HPO, selection and reporting (audit G8).
Attempts are compared on CV over the training split; the holdout is
scored separately and only once per run (audit G7). Latency is measured
in absolute terms — milliseconds per 1 000 predicted rows — so it means
the same thing in every iteration (audit G6).
"""

from __future__ import annotations

import io
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_validate

from .. import config
from ..schemas import MetricSummary, ProblemType, RequirementSpec

CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "roc_auc")
REGRESSION_METRICS = ("r2", "rmse", "mae")
_LOWER_IS_BETTER = frozenset({"rmse", "mae"})


def greater_is_better(metric: str) -> bool:
    return metric not in _LOWER_IS_BETTER


def sklearn_scorer(metric: str, n_classes: int) -> str:
    if metric == "roc_auc":
        return "roc_auc" if n_classes <= 2 else "roc_auc_ovr_weighted"
    return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}.get(metric, metric)


def minority_share(y: pd.Series) -> float:
    counts = y.value_counts(normalize=True)
    return float(counts.min()) if len(counts) else 1.0


def resolve_primary_metric(requirement: RequirementSpec, y: pd.Series) -> tuple[str, bool, list[str]]:
    """Returns (primary_metric, use_balanced_class_weights, notes).

    Classification default: f1_weighted, or f1_macro when the minority class
    is under config.IMBALANCE_MINORITY_SHARE (then class weights are used
    too). Regression default: r2. A requested metric is honoured when it is
    valid for the task (roc_auc only for binary targets).
    """
    notes: list[str] = []
    if requirement.problem_type == "classification":
        imbalanced = y.nunique() > 1 and minority_share(y) < config.IMBALANCE_MINORITY_SHARE
        default = "f1_macro" if imbalanced else "f1_weighted"
        if imbalanced:
            notes.append(
                f"minority class share {minority_share(y):.1%} < {config.IMBALANCE_MINORITY_SHARE:.0%}: "
                "using balanced class weights where supported"
            )
        requested = requirement.primary_metric
        if requested in CLASSIFICATION_METRICS and not (requested == "roc_auc" and y.nunique() > 2):
            return requested, imbalanced, notes
        if requested is not None:
            notes.append(f"primary_metric '{requested}' not valid for this target; used {default}")
        return default, imbalanced, notes

    requested = requirement.primary_metric
    if requested in REGRESSION_METRICS:
        return requested, False, notes
    if requested is not None:
        notes.append(f"primary_metric '{requested}' not valid for regression; used r2")
    return "r2", False, notes


def _scorers(problem_type: ProblemType, n_classes: int) -> dict[str, str]:
    names = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS
    return {m: sklearn_scorer(m, n_classes) for m in names}


def cv_evaluate(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv,
    groups: np.ndarray | None,
    problem_type: ProblemType,
) -> dict[str, MetricSummary]:
    """CV score of a fixed pipeline on training rows. Metrics that cannot be
    computed (e.g. ROC AUC without predict_proba) are omitted."""
    n_classes = int(y.nunique()) if problem_type == "classification" else 0
    scorers = _scorers(problem_type, n_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(pipeline, X, y, cv=cv, groups=groups, scoring=scorers, error_score=np.nan)
    summary: dict[str, MetricSummary] = {}
    for metric in scorers:
        folds = np.asarray(scores[f"test_{metric}"], dtype=float)
        if not greater_is_better(metric):
            folds = -folds
        if np.isnan(folds).any():
            continue
        summary[metric] = MetricSummary(
            mean=float(folds.mean()), std=float(folds.std()), folds=[float(v) for v in folds]
        )
    return summary


def flat_metrics(cv: dict[str, MetricSummary]) -> dict[str, float]:
    """CV means as a flat dict, with the legacy 'f1' alias (= f1_weighted)."""
    flat = {name: summary.mean for name, summary in cv.items()}
    if "f1_weighted" in flat:
        flat["f1"] = flat["f1_weighted"]
    return flat


def compute_metrics(
    pipeline, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType
) -> dict[str, float]:
    y_pred = pipeline.predict(X)
    if problem_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        }
        metrics["f1"] = metrics["f1_weighted"]
        if hasattr(pipeline, "predict_proba") and y.nunique() > 1:
            try:
                proba = pipeline.predict_proba(X)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=pipeline.classes_)
                    )
            except (ValueError, AttributeError):
                pass
        return metrics
    return {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(root_mean_squared_error(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
    }


def measure_latency_ms_per_1k(pipeline, X_sample: pd.DataFrame, repeats: int = config.LATENCY_REPEATS) -> float:
    """Median wall-clock of `predict` over `repeats` runs, scaled to 1 000 rows."""
    if len(X_sample) == 0:
        return 0.0
    timings = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        pipeline.predict(X_sample)
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings) * 1000.0 * (1000.0 / len(X_sample)))


def model_size_bytes(pipeline) -> int | None:
    try:
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        return buffer.tell()
    except Exception:  # sizing is informational only
        return None
