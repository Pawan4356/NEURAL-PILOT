"""Explainability (engine component 8).

Top-10 mean |SHAP| over a small sample. Both the background and the
explained sample are drawn from the training rows, so explanation never
touches the holdout. SHAP failures never fail the run: the model's own
importances (feature_importances_ or |coef_|) are used instead, and
`method` in the summary says which one the report shows.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ..schemas import BlockDiagnostics


def _top(feature_names, values, method: str) -> dict:
    importance = sorted(zip(feature_names, values), key=lambda pair: pair[1], reverse=True)[:10]
    # `mean_abs_shap` keeps its name for either method, so stored reports still read.
    return {
        "method": method,
        "top_features": [{"feature": str(f), "mean_abs_shap": float(v)} for f, v in importance],
    }


def _model_importance(model) -> np.ndarray | None:
    """The estimator's own importances, one per transformed feature, or None."""
    values = getattr(model, "feature_importances_", None)
    if values is None and getattr(model, "coef_", None) is not None:
        values = np.abs(np.asarray(model.coef_, dtype=float))
        if values.ndim > 1:  # one row per class: average across classes
            values = values.mean(axis=0)
    return None if values is None else np.asarray(values, dtype=float).ravel()


def shap_summary(
    pipeline: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, *, seed: int = 42
) -> tuple[dict | None, BlockDiagnostics]:
    start = time.perf_counter()
    feature_names = None
    model = pipeline.steps[-1][1]
    try:
        preprocess = Pipeline(pipeline.steps[:-1])

        background = preprocess.transform(X_background.sample(min(50, len(X_background)), random_state=seed))
        sample = preprocess.transform(X_explain.sample(min(50, len(X_explain)), random_state=seed + 1))
        try:
            feature_names = list(preprocess.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(sample.shape[1])]

        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(sample)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        if mean_abs.ndim > 1:  # multi-class: average across classes
            mean_abs = mean_abs.mean(axis=-1)

        summary = _top(feature_names, mean_abs.tolist(), "shap")
        diagnostics = BlockDiagnostics(
            block="shap", duration_seconds=time.perf_counter() - start, details={"n_features": len(feature_names)}
        )
        return summary, diagnostics
    except Exception as exc:  # SHAP failures never fail the run
        error = str(exc) or type(exc).__name__

    values = _model_importance(model)
    if values is not None:
        names = feature_names if feature_names and len(feature_names) == len(values) else None
        names = names or [f"f{i}" for i in range(len(values))]
        diagnostics = BlockDiagnostics(
            block="shap",
            duration_seconds=time.perf_counter() - start,
            details={"n_features": len(values), "fallback": "model_importance"},
            error=error,
        )
        return _top(names, values.tolist(), "model_importance"), diagnostics

    diagnostics = BlockDiagnostics(
        block="shap", duration_seconds=time.perf_counter() - start, details={}, error=error
    )
    return None, diagnostics
