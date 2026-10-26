"""Explainability (engine component 8).

Top-10 mean |SHAP| over a small sample. Both the background and the
explained sample are drawn from the training rows, so explanation never
touches the holdout. SHAP failures never fail the run.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ..schemas import BlockDiagnostics


def shap_summary(
    pipeline: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, *, seed: int = 42
) -> tuple[dict | None, BlockDiagnostics]:
    start = time.perf_counter()
    try:
        preprocess = Pipeline(pipeline.steps[:-1])
        model = pipeline.steps[-1][1]

        background = preprocess.transform(X_background.sample(min(50, len(X_background)), random_state=seed))
        sample = preprocess.transform(X_explain.sample(min(50, len(X_explain)), random_state=seed + 1))
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
