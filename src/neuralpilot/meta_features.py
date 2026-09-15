"""Dataset meta-feature extraction.

Produces two things from a pandas DataFrame:
- a small numeric vector used for nearest-neighbor retrieval (§5) and
  plateau-independent similarity lookups
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

    def vector(self) -> list[float]:
        """The 5-dim vector used for cosine-similarity retrieval (§5)."""
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
        nunique = series.nunique(dropna=True)
        return nunique <= 20 and nunique / max(len(series), 1) < 0.05
    return False


def extract_meta_features(
    df: pd.DataFrame, target_column: str | None = None
) -> DatasetMetaFeatures:
    n_rows, n_cols = df.shape
    missing_pct = float(df.isna().mean().mean() * 100) if n_cols else 0.0

    categorical_cols = [c for c in df.columns if is_categorical_column(df[c])]
    categorical_ratio = len(categorical_cols) / n_cols if n_cols else 0.0

    n_classes = 0
    if target_column is not None and target_column in df.columns:
        n_classes = int(df[target_column].nunique(dropna=True))

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

    return DatasetMetaFeatures(
        n_rows=n_rows,
        n_cols=n_cols,
        n_classes=n_classes,
        missing_pct=missing_pct,
        categorical_ratio=categorical_ratio,
        columns=columns,
    )
