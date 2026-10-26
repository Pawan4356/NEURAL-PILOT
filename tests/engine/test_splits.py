import numpy as np
import pandas as pd

from neuralpilot.engine import prepare_dataset, run_attempt
from neuralpilot.engine.splits import build_split_spec, cv_splitter, make_holdout_split
from neuralpilot.schemas import ConstraintWeights, PlanSpec, PreprocessingStep, RequirementSpec, TuningStrategy


def _requirement(problem_type="regression", **extra):
    return RequirementSpec(
        target_column="y", target_confidence=1.0, problem_type=problem_type,
        constraint_weights=ConstraintWeights(), weight_confidence=1.0, **extra,
    )


def _time_df(n=200):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    shuffled = rng.permutation(n)
    return pd.DataFrame(
        {
            "date": dates[shuffled].strftime("%Y-%m-%d"),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def test_time_split_puts_only_later_rows_in_holdout():
    df = _time_df()
    spec = build_split_spec(df, _requirement(time_column="date"))
    assert spec.strategy == "time"
    split = make_holdout_split(df, df["y"], spec)
    dates = pd.to_datetime(df["date"])
    assert dates.loc[split.train_index].max() < dates.loc[split.test_index].min()


def test_time_cv_folds_are_forward_chaining():
    df = _time_df()
    prepared = prepare_dataset(df, _requirement(time_column="date"))
    dates = pd.to_datetime(df["date"]).loc[prepared.split.train_index].to_numpy()
    cv = cv_splitter(prepared.split.spec, "regression", prepared.y_train)
    for train_pos, valid_pos in cv.split(prepared.X_train):
        assert dates[train_pos].max() < dates[valid_pos].min()
    # The time column itself is not a feature.
    assert "date" not in prepared.X.columns


def test_group_split_keeps_groups_disjoint():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {
            "patient": rng.integers(0, 40, n).astype(str),
            "x": rng.normal(size=n),
            "y": rng.choice([0, 1], n),
        }
    )
    req = _requirement("classification", group_column="patient")
    prepared = prepare_dataset(df, req)
    assert prepared.split.spec.strategy == "group"
    train_groups = set(df.loc[prepared.split.train_index, "patient"])
    test_groups = set(df.loc[prepared.split.test_index, "patient"])
    assert train_groups.isdisjoint(test_groups)

    groups = prepared.groups.loc[prepared.split.train_index].to_numpy()
    cv = cv_splitter(prepared.split.spec, "classification", prepared.y_train)
    for train_pos, valid_pos in cv.split(prepared.X_train, prepared.y_train, groups):
        assert set(groups[train_pos]).isdisjoint(groups[valid_pos])

    plan = PlanSpec(
        preprocessing=[PreprocessingStep(category="encoder", component="OneHotEncoder")],
        model_family="LinearModel",
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )
    assert run_attempt(plan, prepared).result.attempt.split.strategy == "group"


def test_unknown_time_column_falls_back_with_note():
    notes: list[str] = []
    spec = build_split_spec(_time_df(), _requirement(time_column="nope"), assumptions=notes)
    assert spec.strategy == "random"
    assert notes and "not usable" in notes[0]


def test_classification_default_is_stratified():
    df = pd.DataFrame({"x": range(100), "y": [0] * 80 + [1] * 20})
    spec = build_split_spec(df, _requirement("classification"))
    split = make_holdout_split(df, df["y"], spec)
    assert spec.strategy == "stratified"
    assert df.loc[split.test_index, "y"].mean() == 0.2
