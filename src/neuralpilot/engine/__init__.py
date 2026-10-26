"""Execution engine (§4.2) — deterministic, no LLM.

Split into the nine components of the audit's engine contract:
loader, preprocessing, feature_engineering, model_factory, hpo, training,
evaluation, explainability, tracking — plus `splits` (evaluation protocol)
and `runner`, which composes them into one measured attempt.
"""

from .loader import DatasetLoadError, LoadedDataset, load_dataset
from .runner import (
    AttemptBudget,
    AttemptOutcome,
    PreparedData,
    holdout_evaluate,
    plan_signature,
    prepare_dataset,
    run_attempt,
)

__all__ = [
    "AttemptBudget",
    "AttemptOutcome",
    "DatasetLoadError",
    "LoadedDataset",
    "PreparedData",
    "holdout_evaluate",
    "load_dataset",
    "plan_signature",
    "prepare_dataset",
    "run_attempt",
]
