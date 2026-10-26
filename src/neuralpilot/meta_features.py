"""Dataset meta-feature extraction.

Produces two things from a pandas DataFrame:
- the meta-features used by experiment memory. The scaled retrieval vector
  is built from them in `memory/similarity.py`
- a column-level summary (names/dtypes/cardinality — never raw rows) that
  is safe to hand to an LLM per the Requirement Understanding Agent's
  input contract (§3.1)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass
class DatasetMetaFeatures:
    n_rows: int
    n_cols: int
    n_classes: int  # 0 if target not yet known / not classification
    missing_pct: float
    categorical_ratio: float
    columns: list[dict[str, Any]]
    # Target-aware extras used by the retrieval vector (memory_vector).
    max_cardinality: int = 0  # largest categorical feature cardinality
    minority_class_ratio: float | None = None  # classification-like targets
    target_skew: float | None = None  # numeric targets
    # Meta-features v2 (Phase 7). Records stored before v2 lack these fields
    # and get these defaults, which the similarity vector treats as neutral.
    text_ratio: float = 0.0  # share of high-cardinality free-text features
    datetime_present: bool = False
    mean_abs_skew_numeric: float | None = None  # mean |skew| over numeric features

    def vector(self) -> list[float]:
        """Legacy raw 5-dim vector, kept in stored records for reference.
        Retrieval uses memory_vector(), which is scaled."""
        return [
            float(self.n_rows),
            float(self.n_cols),
            float(self.n_classes),
            float(self.missing_pct),
            float(self.categorical_ratio),
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vector"] = self.vector()
        return data


def is_categorical_column(series: pd.Series) -> bool:
    """True for anything that should go through an encoder rather than a
    scaler: object/string/category dtypes (covers both legacy object-dtype
    strings and pandas >=3's native string dtype) plus low-cardinality
    numeric columns.
    """
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        # low-cardinality numeric columns are treated as categorical-ish
        # As we target both no. of unique values and their ratio w.r.t. size of the series
        nunique = series.nunique(dropna=True)
        return nunique <= 20 and nunique / max(len(series), 1) < 0.05
    return False


def is_text_column(series: pd.Series) -> bool:
    """Free text: string-like values that are mostly distinct. Such a column
    is not a useful category, so it is counted apart from categoricals."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    nunique = series.nunique(dropna=True)
    return nunique > 50 and nunique / max(len(series), 1) > 0.5


def _mean_abs_skew(df: pd.DataFrame, columns: list[str]) -> float | None:
    """Mean absolute skew of the numeric feature columns, or None when there
    are none (or none has a finite skew)."""
    skews = []
    for col in columns:
        skew = df[col].astype(float).skew()
        if pd.notna(skew):
            skews.append(abs(float(skew)))
    return float(sum(skews) / len(skews)) if skews else None


def extract_meta_features(
    df: pd.DataFrame, target_column: str | None = None
) -> DatasetMetaFeatures:
    n_rows, n_cols = df.shape
    # df.isna() - df of same shape with boolean values (true - Null)
    # df.isna().mean() - Column wise
    # df.isna().mean().mean() - Overall avg.
    missing_pct = float(df.isna().mean().mean() * 100) if n_cols else 0.0

    # It gives a quick numerical description of the dataset's structure.
    categorical_cols = [c for c in df.columns if is_categorical_column(df[c])]
    categorical_ratio = len(categorical_cols) / n_cols if n_cols else 0.0

    n_classes = 0
    if target_column is not None and target_column in df.columns:
        n_classes = int(df[target_column].nunique(dropna=True))

    feature_cols = [c for c in df.columns if c != target_column]
    max_cardinality = max(
        (int(df[c].nunique(dropna=True)) for c in feature_cols if c in categorical_cols), default=0
    )
    minority_class_ratio = None
    target_skew = None
    if target_column is not None and target_column in df.columns:
        target = df[target_column].dropna()
        if 1 < target.nunique() <= 50:
            minority_class_ratio = float(target.value_counts(normalize=True).min())
        if pd.api.types.is_numeric_dtype(target) and not pd.api.types.is_bool_dtype(target) and len(target) > 2:
            skew = target.astype(float).skew()
            target_skew = None if pd.isna(skew) else float(skew)

    columns = [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique(dropna=True)),
            "missing_pct": float(df[col].isna().mean() * 100),
            "is_categorical": col in categorical_cols,
        }
        for col in df.columns
    ]

    text_cols = [c for c in feature_cols if is_text_column(df[c])]
    numeric_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    return DatasetMetaFeatures(
        n_rows=n_rows,
        n_cols=n_cols,
        n_classes=n_classes,
        missing_pct=missing_pct,
        categorical_ratio=categorical_ratio,
        columns=columns,
        max_cardinality=max_cardinality,
        minority_class_ratio=minority_class_ratio,
        target_skew=target_skew,
        text_ratio=len(text_cols) / len(feature_cols) if feature_cols else 0.0,
        datetime_present=any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in feature_cols),
        mean_abs_skew_numeric=_mean_abs_skew(df, numeric_cols),
    )

