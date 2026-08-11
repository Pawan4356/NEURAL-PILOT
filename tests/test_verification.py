import numpy as np
import pandas as pd
import pytest

from agentic_automl.schemas import (
    ConstraintWeights,
    PlanSpec,
    PreprocessingStep,
    RequirementSpec,
    TuningStrategy,
)
from agentic_automl.verification import verify_plan


def _requirement(problem_type="classification", target="y"):
    return RequirementSpec(
        target_column=target,
        target_confidence=0.9,
        problem_type=problem_type,
        constraint_weights=ConstraintWeights(),
        weight_confidence=0.9,
    )


def _plan(model_family="RandomForest", encoder="OneHotEncoder", scaler="StandardScaler"):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component=encoder),
            PreprocessingStep(category="scaler", component=scaler),
        ],
        model_family=model_family,
        tuning=TuningStrategy(component="OptunaRandom"),
    )


def _clf_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.choice(["a", "b", "c"], size=n),
            "y": rng.choice([0, 1], size=n),
        }
    )


def test_valid_plan_passes():
    result = verify_plan(_plan(), _requirement(), _clf_df())
    assert result.passed
    assert result.failures == []


def test_missing_target_column_fails():
    result = verify_plan(_plan(), _requirement(target="nope"), _clf_df())
    assert not result.passed
    assert any("does not exist" in f for f in result.failures)


def test_target_over_80pct_missing_blocks():
    df = _clf_df()
    df.loc[df.index[:90], "y"] = np.nan
    result = verify_plan(_plan(), _requirement(), df)
    assert not result.passed
    assert any("missing values" in f for f in result.failures)


def test_classification_single_class_fails():
    df = _clf_df()
    df["y"] = 1
    result = verify_plan(_plan(), _requirement(), df)
    assert not result.passed
    assert any("unique value" in f for f in result.failures)


def test_regression_non_numeric_target_fails():
    df = _clf_df()
    df["y"] = df["x2"]
    result = verify_plan(_plan(), _requirement(problem_type="regression"), df)
    assert not result.passed
    assert any("not numeric" in f for f in result.failures)


def test_class_imbalance_warns_not_blocks():
    n = 220
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.choice(["a", "b"], size=n),
            "y": [0] * 205 + [1] * 15,
        }
    )
    result = verify_plan(_plan(), _requirement(), df)
    assert result.passed
    assert any("imbalance" in w for w in result.warnings)


def test_high_dimensionality_warns_not_blocks():
    n = 20
    rng = np.random.default_rng(2)
    cols = {f"x{i}": rng.normal(size=n) for i in range(10)}
    cols["y"] = rng.choice([0, 1], size=n)
    df = pd.DataFrame(cols)
    result = verify_plan(_plan(), _requirement(), df)
    assert result.passed
    assert any("n_rows/10" in w for w in result.warnings)


def test_unknown_component_triggers_fallback_not_failure():
    result = verify_plan(_plan(model_family="MysteryNet"), _requirement(), _clf_df())
    assert result.passed
    assert any(fb["requested"] == "MysteryNet" and fb["used"] == "RandomForest" for fb in result.fallbacks)
