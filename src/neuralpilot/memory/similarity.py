"""Experiment-memory similarity (Phase 7, Deliverable 7.4).

Each dataset is reduced to a short vector of meta-features, each scaled to
[0, 1] by a fixed, documented transform. The scales are not fitted to the
stored runs, so a similarity score means the same thing as the memory grows.
Similarity is exp(-Euclidean distance) between two such vectors.

Records stored before a feature existed have no value for it. They get a
neutral 0 for that dimension, so they are compared on the features they do
have rather than being dropped.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Names of the memory_vector() dimensions, in order.
MEMORY_VECTOR_FEATURES = (
    "log_rows",
    "log_features",
    "categorical_ratio",
    "missing_ratio",
    "log_max_cardinality",
    "n_classes",
    "target_shape",
    "text_ratio",
    "datetime_present",
    "feature_skew",
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def memory_vector(meta: dict[str, Any], problem_type: str | None = None) -> list[float]:
    """Scaled meta-feature vector for experiment-memory retrieval.

    - log_rows            log10(n_rows) / 7           (10 rows .. 10M rows)
    - log_features        log10(n_features) / 4       (1 .. 10k features)
    - categorical_ratio   share of categorical columns
    - missing_ratio       missing_pct / 100
    - log_max_cardinality log10(max categorical cardinality + 1) / 5
    - n_classes           min(n_classes, 20) / 20     (0 for regression)
    - target_shape        classification: 2 x minority class share;
                          regression: min(|target skew| / 5, 1)
    - text_ratio          share of free-text columns
    - datetime_present    1 when any feature is a date/time, else 0
    - feature_skew        min(mean |skew| of numeric features / 3, 1)

    Works on stored dicts from older records too: missing v2 fields count as
    neutral (0), and the rest of the vector is read as before.
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

    feature_skew = meta.get("mean_abs_skew_numeric")
    skew_dim = _clip01(float(feature_skew) / 3.0) if feature_skew is not None else 0.0

    return [
        _clip01(math.log10(n_rows) / 7.0),
        _clip01(math.log10(n_features) / 4.0),
        _clip01(meta.get("categorical_ratio") or 0.0),
        _clip01((meta.get("missing_pct") or 0.0) / 100.0),
        _clip01(math.log10(max_cardinality + 1) / 5.0),
        class_dim,
        shape_dim,
        _clip01(meta.get("text_ratio") or 0.0),
        1.0 if meta.get("datetime_present") else 0.0,
        skew_dim,
    ]


def similarity(query: list[float], candidate: list[float]) -> float:
    """exp(-distance) in (0, 1]; 1 means identical meta-features."""
    distance = float(np.linalg.norm(np.asarray(query, dtype=float) - np.asarray(candidate, dtype=float)))
    return math.exp(-distance)
