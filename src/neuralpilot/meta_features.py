"""Dataset meta-feature extraction.

Produces two things from a pandas DataFrame:
- a small numeric vector used for nearest-neighbor retrieval (§5) and
  plateau-independent similarity lookups
- a column-level summary (names/dtypes/cardinality — never raw rows) that
  is safe to hand to an LLM per the Requirement Understanding Agent's
  input contract (§3.1)
"""

from __future__ import annotations

import math
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
    )


# Names of the memory_vector() dimensions, in order.
MEMORY_VECTOR_FEATURES = (
    "log_rows",
    "log_features",
    "categorical_ratio",
    "missing_ratio",
    "log_max_cardinality",
    "n_classes",
    "target_shape",
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def memory_vector(meta: dict[str, Any], problem_type: str | None = None) -> list[float]:
    """Scaled meta-feature vector for experiment-memory retrieval.

    Each dimension is mapped to [0, 1] with a fixed, documented transform
    (not fitted to the stored runs, so scaling never drifts as memory
    grows). Raw row counts no longer dominate the distance:
    - log_rows            log10(n_rows) / 7           (10 rows .. 10M rows)
    - log_features        log10(n_features) / 4       (1 .. 10k features)
    - categorical_ratio   share of categorical columns
    - missing_ratio       missing_pct / 100
    - log_max_cardinality log10(max categorical cardinality + 1) / 5
    - n_classes           min(n_classes, 20) / 20     (0 for regression)
    - target_shape        classification: 2 x minority class share;
                          regression: min(|target skew| / 5, 1)

    Works on stored dicts from older records too, using neutral defaults
    for fields they do not have.
    """
    n_rows = max(int(meta.get("n_rows") or 1), 1)
    n_features = max(int(meta.get("n_cols") or 2) - 1, 1)
    max_cardinality = meta.get("max_cardinality")
    if max_cardinality is None:  # older records: derive from the column summary
        max_cardinality = max(
            (int(c.get("n_unique", 0)) for c in meta.get("columns", []) if c.get("is_categorical")), default=0
        )
    n_classes = int(meta.get("n_classes") or 0)

    if problem_type == "regression":
        class_dim = 0.0
        skew = meta.get("target_skew")
        shape_dim = _clip01(abs(skew) / 5.0) if skew is not None else 0.0
    else:
        class_dim = _clip01(min(n_classes, 20) / 20.0)
        minority = meta.get("minority_class_ratio")
        if minority is None:
            minority = 1.0 / n_classes if n_classes >= 2 else 0.5
        shape_dim = _clip01(2.0 * float(minority))

    return [
        _clip01(math.log10(n_rows) / 7.0),
        _clip01(math.log10(n_features) / 4.0),
        _clip01(meta.get("categorical_ratio") or 0.0),
        _clip01((meta.get("missing_pct") or 0.0) / 100.0),
        _clip01(math.log10(max_cardinality + 1) / 5.0),
        class_dim,
        shape_dim,
    ]
