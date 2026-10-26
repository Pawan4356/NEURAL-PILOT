"""Data splitting (evaluation protocol).

One holdout split is made per run and reused by every attempt, so
attempts are compared on cross-validation over the training part only and
the holdout is scored exactly once, at finalisation (audit G7).

Strategies:
- stratified: classification default; class ratios kept in train/test/folds
- random: regression default
- group: rows sharing a group id never straddle train/test or folds
- time: train is strictly earlier than test; folds are forward-chaining
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from .. import config
from ..schemas import ProblemType, RequirementSpec, SplitSpec


@dataclass
class DataSplit:
    spec: SplitSpec
    train_index: pd.Index
    test_index: pd.Index


def build_split_spec(
    df: pd.DataFrame,
    requirement: RequirementSpec,
    *,
    seed: int = config.SEED,
    assumptions: list[str] | None = None,
) -> SplitSpec:
    """Pick a strategy from the requirement. Time/group columns are only used
    when the requirement names them and they exist in the data."""
    notes = assumptions if assumptions is not None else []
    default = "stratified" if requirement.problem_type == "classification" else "random"
    common = dict(
        test_size=config.HOLDOUT_FRACTION,
        n_folds=config.CV_FOLDS,
        seed=seed,
        hpo_max_rows=config.HPO_MAX_ROWS,
    )
    if requirement.time_column:
        if requirement.time_column in df.columns and requirement.time_column != requirement.target_column:
            return SplitSpec(strategy="time", time_column=requirement.time_column, **common)
        notes.append(f"time_column '{requirement.time_column}' not usable; used {default} split instead")
    if requirement.group_column:
        if requirement.group_column in df.columns and requirement.group_column != requirement.target_column:
            return SplitSpec(strategy="group", group_column=requirement.group_column, **common)
        notes.append(f"group_column '{requirement.group_column}' not usable; used {default} split instead")
    return SplitSpec(strategy=default, **common)


def _time_order(df: pd.DataFrame, column: str) -> pd.Index:
    values = df[column]
    if not pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_datetime64_any_dtype(values):
        values = pd.to_datetime(values, errors="coerce", format="mixed")
    return values.sort_values(kind="stable", na_position="first").index


def make_holdout_split(df: pd.DataFrame, y: pd.Series, spec: SplitSpec) -> DataSplit:
    """Split row labels of `df` into train/test according to `spec`."""
    index = df.index
    if spec.strategy == "time":
        ordered = _time_order(df, spec.time_column)
        n_test = max(1, int(round(len(ordered) * spec.test_size)))
        return DataSplit(spec, ordered[:-n_test], ordered[-n_test:])
    if spec.strategy == "group":
        groups = df[spec.group_column].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=spec.test_size, random_state=spec.seed)
        train_pos, test_pos = next(splitter.split(np.zeros(len(index)), groups=groups))
        return DataSplit(spec, index[train_pos], index[test_pos])
    stratify = None
    if spec.strategy == "stratified" and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y
    train_idx, test_idx = train_test_split(
        index, test_size=spec.test_size, random_state=spec.seed, stratify=stratify
    )
    return DataSplit(spec, pd.Index(train_idx), pd.Index(test_idx))


def subsample_for_hpo(
    X: pd.DataFrame, y: pd.Series, spec: SplitSpec, problem_type: ProblemType
) -> tuple[pd.DataFrame, pd.Series]:
    """Bound HPO/CV cost on large data. Keeps class ratios (stratified),
    time order (time) — the full training set is still used for the final fit."""
    limit = spec.hpo_max_rows
    if limit is None or len(X) <= limit:
        return X, y
    if spec.strategy == "time":
        return X.iloc[-limit:], y.iloc[-limit:]
    stratify = None
    if problem_type == "classification" and y.value_counts().min() >= 2:
        stratify = y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=limit, random_state=spec.seed, stratify=stratify
    )
    return X_sub, y_sub


def cv_splitter(spec: SplitSpec, problem_type: ProblemType, y: pd.Series):
    """Return a CV splitter for the training part only."""
    if spec.strategy == "time":
        return TimeSeriesSplit(n_splits=spec.n_folds)
    if spec.strategy == "group":
        return GroupKFold(n_splits=spec.n_folds)
    if problem_type == "classification" and y.nunique() > 1:
        n_splits = max(2, min(spec.n_folds, int(y.value_counts().min())))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=spec.seed)
    return KFold(n_splits=spec.n_folds, shuffle=True, random_state=spec.seed)


def group_labels(df: pd.DataFrame, spec: SplitSpec) -> pd.Series | None:
    """Group ids aligned to `df`'s index (kept apart from the feature matrix,
    since group ids are usually identifiers that must not be features)."""
    if spec.strategy != "group":
        return None
    return df[spec.group_column].astype(str)
