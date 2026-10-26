import numpy as np
import pandas as pd
import pytest

from neuralpilot import config
from neuralpilot.engine import AttemptBudget, plan_signature, prepare_dataset, run_attempt
from neuralpilot.engine import runner as engine_runner
from neuralpilot.engine.evaluation import sklearn_scorer
from neuralpilot.engine.feature_engineering import build_fe_steps
from neuralpilot.engine.preprocessing import build_preprocessor, validate_params
from neuralpilot.execution import run_pipeline
from neuralpilot.schemas import (
    ConstraintWeights,
    FeatureEngineeringStep,
    PlanSpec,
    PreprocessingStep,
    RequirementSpec,
    TuningStrategy,
)


def _clf_df(n=240, seed=0, minority=None):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b", "c"], size=n)
    if minority is None:
        y = (x1 + (x2 == "a") + rng.normal(scale=0.5, size=n) > 0.3).astype(int)
    else:
        y = (rng.random(n) < minority).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": rng.normal(size=n), "y": y})


def _requirement(problem_type="classification", **extra):
    return RequirementSpec(
        target_column="y", target_confidence=1.0, problem_type=problem_type,
        constraint_weights=ConstraintWeights(), weight_confidence=1.0, **extra,
    )


def _plan(model="RandomForest", tuner="NoTuning", n_trials=0, fe=None, encoder_params=None):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder", params=encoder_params or {}),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        feature_engineering=fe or [],
        model_family=model,
        tuning=TuningStrategy(component=tuner, n_trials=n_trials),
    )


# --- determinism / wrapper parity (golden) ---------------------------------


def test_same_seed_gives_identical_cv_metrics():
    df = _clf_df()
    prepared = prepare_dataset(df, _requirement())
    plan = _plan(model="LinearModel", tuner="OptunaTPE", n_trials=3)
    first = run_attempt(plan, prepared)
    second = run_attempt(plan, prepared)
    assert first.result.metrics == second.result.metrics
    assert first.result.attempt.best_params == second.result.attempt.best_params


def test_run_pipeline_wrapper_matches_engine_attempt():
    df = _clf_df()
    wrapped = run_pipeline(_plan(), _requirement(), df)
    direct = run_attempt(_plan(), prepare_dataset(df, _requirement()))
    assert wrapped.result.attempt.cv == direct.result.attempt.cv
    # The wrapper reports holdout metrics; the attempt reports CV means.
    assert wrapped.result.metrics != direct.result.metrics
    assert any(d.block == "holdout" for d in wrapped.result.diagnostics)


# --- holdout isolation ------------------------------------------------------


def test_holdout_rows_never_reach_hpo_or_cv(monkeypatch):
    seen: list[pd.Index] = []
    real_tune = engine_runner.hpo.tune
    real_cv = engine_runner.evaluation.cv_evaluate

    def spy_tune(*args, X, **kwargs):
        seen.append(X.index)
        return real_tune(*args, X=X, **kwargs)

    def spy_cv(pipeline, X, *args, **kwargs):
        seen.append(X.index)
        return real_cv(pipeline, X, *args, **kwargs)

    monkeypatch.setattr(engine_runner.hpo, "tune", spy_tune)
    monkeypatch.setattr(engine_runner.evaluation, "cv_evaluate", spy_cv)

    prepared = prepare_dataset(_clf_df(), _requirement())
    run_attempt(_plan(model="LinearModel", tuner="OptunaRandom", n_trials=2), prepared)

    assert len(seen) == 2
    holdout = set(prepared.split.test_index)
    for index in seen:
        assert holdout.isdisjoint(index)


# --- primary metric alignment (audit G8) -------------------------------------


def test_hpo_objective_is_the_primary_metric(monkeypatch):
    captured = {}
    real_tune = engine_runner.hpo.tune

    def spy(*args, scorer_name, **kwargs):
        captured["scorer"] = scorer_name
        return real_tune(*args, scorer_name=scorer_name, **kwargs)

    monkeypatch.setattr(engine_runner.hpo, "tune", spy)
    prepared = prepare_dataset(_clf_df(), _requirement())
    outcome = run_attempt(_plan(model="LinearModel", tuner="OptunaRandom", n_trials=2), prepared)

    assert prepared.primary_metric == "f1_weighted"
    assert captured["scorer"] == sklearn_scorer("f1_weighted", 2)
    assert outcome.result.attempt.primary_metric == "f1_weighted"
    tuning = next(d for d in outcome.result.diagnostics if d.block == "hyperparameters")
    assert tuning.details["objective"] == "f1_weighted"


def test_requested_metric_is_honoured_and_invalid_one_replaced():
    df = _clf_df()
    assert prepare_dataset(df, _requirement(primary_metric="roc_auc")).primary_metric == "roc_auc"

    multiclass = df.assign(y=np.random.default_rng(0).choice(["a", "b", "c"], len(df)))
    prepared = prepare_dataset(multiclass, _requirement(primary_metric="roc_auc"))
    assert prepared.primary_metric == "f1_weighted"
    assert any("not valid" in note for note in prepared.notes)


def test_imbalanced_target_uses_macro_metric_and_class_weights():
    prepared = prepare_dataset(_clf_df(n=600, minority=0.06), _requirement())
    assert prepared.primary_metric == "f1_macro"
    assert prepared.balanced_class_weights
    outcome = run_attempt(_plan(), prepared)
    assert outcome.result.attempt.class_weight == "balanced"
    assert outcome.pipeline.named_steps["model"].class_weight == "balanced"


# --- budgets and bounds (audit G9) -------------------------------------------


def test_n_trials_is_clamped_and_recorded():
    prepared = prepare_dataset(_clf_df(), _requirement())
    outcome = run_attempt(
        _plan(model="LinearModel", tuner="OptunaRandom", n_trials=500), prepared, budget=AttemptBudget(max_trials=2)
    )
    tuning = next(d for d in outcome.result.diagnostics if d.block == "hyperparameters")
    assert tuning.details["n_trials"] == 2
    assert any("clamped to 2" in note for note in outcome.result.plan_adjustments)


def test_hpo_timeout_stops_the_study():
    prepared = prepare_dataset(_clf_df(), _requirement())
    outcome = run_attempt(
        _plan(model="LinearModel", tuner="OptunaRandom", n_trials=50),
        prepared,
        budget=AttemptBudget(max_trials=50, hpo_timeout_s=0.01),
    )
    assert outcome.result.attempt.n_trials_completed < 50
    assert "timeout" in outcome.result.attempt.hpo_stopped_reason


def test_hpo_subsamples_large_training_sets(monkeypatch):
    monkeypatch.setattr(config, "HPO_MAX_ROWS", 100)
    prepared = prepare_dataset(_clf_df(n=400), _requirement())
    outcome = run_attempt(_plan(), prepared)
    assert outcome.result.attempt.hpo_rows == 100
    # Final fit still uses the full training split.
    split_diag = next(d for d in outcome.result.diagnostics if d.block == "split")
    assert split_diag.details["n_train"] == 320


def test_feature_engineering_params_are_bounded():
    plan = _plan(
        fe=[
            FeatureEngineeringStep(name="PolynomialFeatures", params={"degree": 5}),
            FeatureEngineeringStep(name="SelectKBest", params={"k": 1000}),
            FeatureEngineeringStep(name="PCA", params={"n_components": 3.0}),
        ]
    )
    steps, warnings, notes = build_fe_steps(plan, "classification", encoded_width=6)
    poly = steps[0][1]
    assert poly.degree == 2
    assert steps[1][1].k == "all"
    assert steps[2][1].n_components == 0.95
    assert any("clamped" in n for n in notes)
    assert any("keeping all" in n for n in notes)


def test_polynomial_features_dropped_on_wide_data():
    plan = _plan(fe=[FeatureEngineeringStep(name="PolynomialFeatures", params={"degree": 2})])
    steps, _, notes = build_fe_steps(plan, "classification", encoded_width=500)
    assert steps == []
    assert any("dropped" in n for n in notes)


def test_preprocessing_params_are_validated_and_passed_through():
    clean, notes = validate_params("OneHotEncoder", {"max_categories": 2, "sparse_output": True, "min_frequency": -3})
    assert clean == {"max_categories": 2}
    assert len(notes) == 2

    preprocessor, _, adjustments = build_preprocessor(
        _plan(encoder_params={"max_categories": 2}), ["x2"], ["x1"], categorical_cardinality=3
    )
    encoder = {name: t for name, t, _ in preprocessor.transformers}["categorical"].named_steps["encoder"]
    assert encoder.max_categories == 2
    assert adjustments == []


# --- fallback model is tuned with its own search space -----------------------


def test_fallback_model_uses_resolved_search_space():
    prepared = prepare_dataset(_clf_df(), _requirement())
    outcome = run_attempt(_plan(model="MysteryNet", tuner="OptunaRandom", n_trials=1), prepared)
    assert outcome.result.attempt.model_family == "RandomForest"
    assert set(outcome.result.attempt.best_params) == {"n_estimators", "max_depth"}


# --- resources / signature ----------------------------------------------------


def test_attempt_reports_resources_and_cv_summary():
    outcome = run_attempt(_plan(), prepare_dataset(_clf_df(), _requirement()))
    attempt = outcome.result.attempt
    assert attempt.resources.latency_ms_per_1k > 0
    assert attempt.resources.model_bytes and attempt.resources.model_bytes > 0
    summary = attempt.cv["f1_weighted"]
    assert len(summary.folds) == 3 and summary.std >= 0
    assert {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc"} <= set(attempt.cv)


def test_plan_signature_reflects_what_executes():
    base = plan_signature(_plan())
    assert plan_signature(_plan()) == base
    assert plan_signature(_plan(model="LinearModel")) != base
    # An unknown model resolves to the RandomForest fallback, so it is the same execution.
    assert plan_signature(_plan(model="MysteryNet")) == base


def test_identifier_columns_never_reach_the_model():
    df = _clf_df()
    df.insert(0, "customer_id", np.random.default_rng(1).permutation(len(df)))
    prepared = prepare_dataset(df, _requirement())
    outcome = run_attempt(_plan(), prepared)
    prep = next(d for d in outcome.result.diagnostics if d.block == "preprocessing")
    assert "customer_id" in prep.details["excluded_columns"]
    assert "customer_id" not in prep.details["numeric_cols"] + prep.details["categorical_cols"]


@pytest.mark.parametrize("action, kept", [("drop", False), ("warn", True)])
def test_leakage_action(action, kept):
    df = _clf_df(n=400)
    df["leak"] = df["y"] + np.random.default_rng(2).normal(scale=0.01, size=len(df))
    prepared = prepare_dataset(df, _requirement(), leakage_action=action)
    assert ("leak" in prepared.X.columns) is kept
    if kept:
        assert any("possible leakage" in n for n in prepared.notes)
    else:
        assert "suspected leakage" in prepared.excluded_columns["leak"]
