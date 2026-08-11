import pytest

from agentic_automl.schemas import ConstraintWeights
from agentic_automl.scoring import compute_validation, should_generate_shap


def test_classification_composite_score():
    report = compute_validation(
        raw_metrics={"accuracy": 0.9, "f1": 0.88},
        latency_seconds=1.0,
        rolling_max_latency=2.0,
        model_family="XGBoost",
        problem_type="classification",
        weights=ConstraintWeights(accuracy=0.7, latency=0.15, interpretability=0.15),
        per_block_diagnostics=[],
    )
    assert report.accuracy_norm == pytest.approx(0.88)
    assert report.latency_norm == pytest.approx(0.5)
    assert report.interp_norm == pytest.approx(0.4)  # xgboost
    expected = 0.7 * 0.88 + 0.15 * (1 - 0.5) + 0.15 * 0.4
    assert report.composite_score == pytest.approx(expected)


def test_regression_uses_r2_clipped():
    report = compute_validation(
        raw_metrics={"r2": -0.5, "rmse": 3.0},
        latency_seconds=0.5,
        rolling_max_latency=1.0,
        model_family="LinearModel",
        problem_type="regression",
        weights=ConstraintWeights(),
        per_block_diagnostics=[],
    )
    assert report.accuracy_norm == 0.0  # clipped from -0.5
    assert report.interp_norm == pytest.approx(1.0)  # linear


def test_zero_rolling_max_latency_is_safe():
    report = compute_validation(
        raw_metrics={"accuracy": 0.5},
        latency_seconds=0.0,
        rolling_max_latency=0.0,
        model_family="RandomForest",
        problem_type="classification",
        weights=ConstraintWeights(),
        per_block_diagnostics=[],
    )
    assert report.latency_norm == 0.0


def test_unknown_model_family_uses_default_interp():
    report = compute_validation(
        raw_metrics={"accuracy": 0.5},
        latency_seconds=1.0,
        rolling_max_latency=1.0,
        model_family="NotReal",
        problem_type="classification",
        weights=ConstraintWeights(),
        per_block_diagnostics=[],
    )
    assert report.interp_norm == 0.5


def test_weights_are_normalized():
    report = compute_validation(
        raw_metrics={"accuracy": 1.0},
        latency_seconds=0.0,
        rolling_max_latency=1.0,
        model_family="DecisionTree",
        problem_type="classification",
        weights=ConstraintWeights(accuracy=7, latency=1.5, interpretability=1.5),
        per_block_diagnostics=[],
    )
    # accuracy_norm=1, latency_norm=0 -> (1-0)=1, interp_norm=1 (tree)
    # composite should equal 1.0 regardless of un-normalized weight scale
    assert report.composite_score == pytest.approx(1.0)


def test_should_generate_shap_explicit_request():
    weights = ConstraintWeights(accuracy=0.4, latency=0.1, interpretability=0.5)
    assert should_generate_shap(weights, weight_confidence=0.9)


def test_should_generate_shap_default_weights_false():
    weights = ConstraintWeights()  # defaults
    assert not should_generate_shap(weights, weight_confidence=0.9)


def test_should_generate_shap_low_confidence_false():
    weights = ConstraintWeights(accuracy=0.4, latency=0.1, interpretability=0.5)
    assert not should_generate_shap(weights, weight_confidence=0.5)
