"""Column profiling and leakage detection (audit G2) — deterministic, no LLM.

`profile_columns` assigns every column a role so the engine can keep
identifiers, constants, raw timestamps and free text out of the model.
`leakage_scan` fits a tiny single-feature model per column and flags any
column that predicts the target almost perfectly on its own — the classic
signature of a leaked label or a post-outcome field.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config
from .meta_features import is_categorical_column

ColumnRole = Literal["id", "constant", "datetime", "text", "categorical", "numeric"]

# Roles that never reach the model as features.
EXCLUDED_ROLES: frozenset[str] = frozenset({"id", "constant", "datetime", "text"})

# "id", "customer_id", "Customer Id", "order-key", "uuid" ... (case-insensitive)
_ID_NAME_SEPARATED = re.compile(r"(^|[_\s\-.])(id|uuid|guid|key)$", re.IGNORECASE)
# "CustomerID", "customerId" — camel-case suffix; deliberately case-sensitive
# so words that merely end in "id" ("paid", "valid", "covid") don't match.
_ID_NAME_CAMEL = re.compile(r"[a-z](ID|Id)$")
_DATE_HINT = re.compile(r"[-/:]")


@dataclass
class ColumnProfile:
    name: str
    role: ColumnRole
    dtype: str
    n_unique: int
    unique_ratio: float
    missing_pct: float
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeakageFlag:
    column: str
    metric: str
    score: float
    baseline: float | None = None

    def describe(self) -> str:
        base = f" (baseline {self.baseline:.3f})" if self.baseline is not None else ""
        return f"'{self.column}' alone predicts the target with {self.metric}={self.score:.3f}{base}"

    def to_dict(self) -> dict:
        return asdict(self)


def _is_integer_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        values = series.dropna()
        return not values.empty and bool(np.all(np.isclose(values, np.round(values))))
    return False


def _is_text_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _id_reason(name: str, series: pd.Series, unique_ratio: float) -> str | None:
    if unique_ratio < config.ID_UNIQUE_RATIO:
        return None
    integer_like = _is_integer_like(series)
    textual = _is_text_dtype(series)
    if (integer_like or textual) and (_ID_NAME_SEPARATED.search(name) or _ID_NAME_CAMEL.search(name)):
        return "identifier name with unique values"
    if integer_like:
        values = series.dropna()
        if values.is_monotonic_increasing or values.is_monotonic_decreasing:
            return "unique integers that follow row order"
    if textual:
        sample = series.dropna().astype(str).head(500)
        if len(sample) >= 20 and not sample.str.contains(" ", regex=False).any():
            return "unique single-token strings (identifier-like)"
    return None


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _is_text_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty or sample.str.contains(_DATE_HINT).mean() < 0.95:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.95)


def _looks_like_text(series: pd.Series, unique_ratio: float) -> bool:
    if not _is_text_dtype(series) or unique_ratio < 0.5:
        return False
    sample = series.dropna().astype(str).head(500)
    return bool(not sample.empty and sample.str.split().str.len().median() >= 3)


def profile_columns(df: pd.DataFrame, target_column: str | None = None) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())
        unique_ratio = n_unique / max(non_null, 1)
        missing_pct = float(series.isna().mean() * 100) if n_rows else 0.0
        base = dict(
            name=col, dtype=str(series.dtype), n_unique=n_unique,
            unique_ratio=float(unique_ratio), missing_pct=missing_pct,
        )

        if col == target_column:
            role: ColumnRole = "categorical" if is_categorical_column(series) else "numeric"
            profiles.append(ColumnProfile(role=role, reason="target", **base))
            continue
        if n_unique <= 1:
            profiles.append(ColumnProfile(role="constant", reason="single distinct value", **base))
            continue
        # Before the ID rule: unique date strings would otherwise look like identifiers.
        if _looks_like_datetime(series):
            profiles.append(
                ColumnProfile(role="datetime", reason="raw timestamp (date feature extraction not supported yet)", **base)
            )
            continue
        id_reason = _id_reason(str(col), series, unique_ratio)
        if id_reason:
            profiles.append(ColumnProfile(role="id", reason=id_reason, **base))
            continue
        if _looks_like_text(series, unique_ratio):
            profiles.append(ColumnProfile(role="text", reason="free text (text features not supported yet)", **base))
            continue
        role = "categorical" if is_categorical_column(series) else "numeric"
        profiles.append(ColumnProfile(role=role, **base))
    return profiles


def _single_column_matrix(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = series.astype(float)
        return values.fillna(values.median() if values.notna().any() else 0.0).to_numpy().reshape(-1, 1)
    codes, _ = pd.factorize(series, use_na_sentinel=True)
    return codes.reshape(-1, 1).astype(float)


def leakage_scan(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    *,
    columns: list[str] | None = None,
    sample: int = config.LEAKAGE_SAMPLE_ROWS,
    seed: int = config.SEED,
) -> list[LeakageFlag]:
    """Flag columns that predict the target almost perfectly on their own.

    A depth-3 tree is fitted per column with 3-fold CV on a sample. Flags:
    binary classification AUC >= threshold; multi-class accuracy >= threshold
    and above the base rate by a margin; regression R² >= threshold, where
    R² is the larger of the tree's CV R² and the squared Spearman
    correlation (a depth-3 tree has only 8 leaves, so it cannot reach a high
    R² on a continuous copy of the target by itself).
    """
    data = df.dropna(subset=[target_column])
    if len(data) > sample:
        data = data.sample(n=sample, random_state=seed)
    if len(data) < 30:
        return []

    y_raw = data[target_column]
    candidates = columns if columns is not None else [c for c in data.columns if c != target_column]
    flags: list[LeakageFlag] = []

    if problem_type == "classification":
        y, _ = pd.factorize(y_raw)
        counts = np.bincount(y)
        if len(counts) < 2 or counts.min() < 3:
            return []
        n_splits = min(3, int(counts.min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        binary = len(counts) == 2
        base_rate = float(counts.max() / counts.sum())
        for col in candidates:
            X = _single_column_matrix(data[col])
            model = DecisionTreeClassifier(max_depth=3, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if binary:
                    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
                    score = float(roc_auc_score(y, proba))
                    if score >= config.LEAKAGE_SCORE_THRESHOLD:
                        flags.append(LeakageFlag(col, "roc_auc", score, 0.5))
                else:
                    pred = cross_val_predict(model, X, y, cv=cv)
                    score = float(accuracy_score(y, pred))
                    if (
                        score >= config.LEAKAGE_SCORE_THRESHOLD
                        and score > base_rate + config.LEAKAGE_ACCURACY_MARGIN
                    ):
                        flags.append(LeakageFlag(col, "accuracy", score, base_rate))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        for col in candidates:
            X = _single_column_matrix(data[col])
            model = DecisionTreeRegressor(max_depth=3, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = cross_val_predict(model, X, y, cv=cv)
                rho = pd.Series(X[:, 0]).corr(pd.Series(y), method="spearman")
            score = max(float(r2_score(y, pred)), 0.0 if pd.isna(rho) else float(rho) ** 2)
            if score >= config.LEAKAGE_SCORE_THRESHOLD:
                flags.append(LeakageFlag(col, "r2", score, 0.0))
    return flags


def derive_problem_type(series: pd.Series) -> Literal["classification", "regression"]:
    """Deterministic task inference for a known target column.

    Non-numeric or boolean targets, numeric targets with at most 20 distinct
    values, and integer targets with at most 50 distinct values that make up
    under 5% of rows are classification; everything else is regression.
    """
    values = series.dropna()
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return "classification"
    n_unique = int(values.nunique())
    if n_unique <= 20:
        return "classification"
    if _is_integer_like(values) and n_unique <= 50 and n_unique / max(len(values), 1) < 0.05:
        return "classification"
    return "regression"
