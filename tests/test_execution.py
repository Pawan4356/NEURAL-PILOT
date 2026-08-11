import numpy as np
import pandas as pd
import pytest

from agentic_automl.schemas import (
    ConstraintWeights,
    FeatureEngineeringStep,
    PlanSpec,
    PreprocessingStep,
    RequirementSpec,
    TuningStrategy,
)
from agentic_automl.execution import run_pipeline


def _clf_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b", "c"], size=n)
    x3 = rng.normal(size=n)
    y = (x1 + (x2 == "a").astype(float) + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    df.loc[df.index[:5], "x1"] = np.nan  # exercise imputation
    return df


def _reg_df(n=150, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b"], size=n)
    y = 2 * x1 + (x2 == "a").astype(float) * 3 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _requirement(problem_type):
    return RequirementSpec(
        target_column="y",
        target_confidence=0.9,
        problem_type=problem_type,
        constraint_weights=ConstraintWeights(),
        weight_confidence=0.9,
    )


def _plan(model_family="RandomForest", n_trials=0, feature_engineering=None):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        feature_engineering=feature_engineering or [],
        model_family=model_family,
        tuning=TuningStrategy(component="NoTuning" if n_trials == 0 else "OptunaRandom", n_trials=n_trials),
    )


def test_classification_pipeline_runs_end_to_end():
    outcome = run_pipeline(_plan(), _requirement("classification"), _clf_df())
    assert "accuracy" in outcome.result.metrics and "f1" in outcome.result.metrics
    assert 0.0 <= outcome.result.metrics["accuracy"] <= 1.0
    assert outcome.result.latency_seconds >= 0
    block_names = {d.block for d in outcome.result.diagnostics}
    assert {"split", "preprocessing", "feature_engineering", "hyperparameters", "model_family", "evaluation"} <= block_names


def test_regression_pipeline_runs_end_to_end():
    outcome = run_pipeline(_plan(model_family="LinearModel"), _requirement("regression"), _reg_df())
    assert "r2" in outcome.result.metrics and "rmse" in outcome.result.metrics


def test_unknown_model_family_falls_back_and_still_runs():
    outcome = run_pipeline(_plan(model_family="MysteryNet"), _requirement("classification"), _clf_df())
    assert any(fb["requested"] == "MysteryNet" for fb in outcome.result.fallbacks)
    assert "accuracy" in outcome.result.metrics


def test_hyperparameter_tuning_runs_with_small_trial_budget():
    outcome = run_pipeline(_plan(n_trials=2), _requirement("classification"), _clf_df())
    tuning_diag = next(d for d in outcome.result.diagnostics if d.block == "hyperparameters")
    assert tuning_diag.details["n_trials"] == 2


def test_unknown_feature_engineering_step_is_skipped_not_fatal():
    plan = _plan(feature_engineering=[FeatureEngineeringStep(name="NotARealStep")])
    outcome = run_pipeline(plan, _requirement("classification"), _clf_df())
    fe_diag = next(d for d in outcome.result.diagnostics if d.block == "feature_engineering")
    assert "NotARealStep" in fe_diag.details["warnings"][0]
    assert "accuracy" in outcome.result.metrics


def test_shap_generation_when_requested():
    outcome = run_pipeline(_plan(), _requirement("classification"), _clf_df(), generate_shap=True)
    assert outcome.result.shap_summary is not None
    assert "top_features" in outcome.result.shap_summary
    assert len(outcome.result.shap_summary["top_features"]) > 0


def test_no_shap_when_not_requested():
    outcome = run_pipeline(_plan(), _requirement("classification"), _clf_df(), generate_shap=False)
    assert outcome.result.shap_summary is None


def test_string_classification_target_is_handled():
    df = _clf_df()
    df["y"] = df["y"].map({0: "no", 1: "yes"})
    outcome = run_pipeline(_plan(), _requirement("classification"), df)
    assert "accuracy" in outcome.result.metrics
