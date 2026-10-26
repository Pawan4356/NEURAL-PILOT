"""Training (engine component 6): the final refit on the full training
split (never the holdout), timed for ResourceUsage."""

from __future__ import annotations

import time
import warnings

import pandas as pd
from sklearn.pipeline import Pipeline


def fit_final(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Fit in place; returns training seconds."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pipeline.fit(X_train, y_train)
    return time.perf_counter() - t0


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far (Linux/macOS)."""
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KiB on Linux, bytes on macOS.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except (ImportError, OSError):
        return None
