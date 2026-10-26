"""Execution Engine (§4.2) — compatibility wrapper.

The implementation lives in the `engine` package. `run_pipeline` keeps the
original single-call API (prepare -> one attempt -> holdout score) for
callers and tests that want one plan measured end to end. The orchestrator
uses the engine directly so it can compare attempts on CV and score the
holdout only once per run.
"""

from __future__ import annotations

import time
import pandas as pd

from . import config
from .engine import AttemptOutcome, prepare_dataset, run_attempt
from .engine.progress import ProgressCallback
from .engine.runner import holdout_evaluate
from .schemas import BlockDiagnostics, PlanSpec, RequirementSpec

# Kept for backwards compatibility with code that imported the old name.
ExecutionOutcome = AttemptOutcome


def run_pipeline(
    plan: PlanSpec,
    requirement: RequirementSpec,
    df: pd.DataFrame,
    *,
    generate_shap: bool = False,
    random_state: int = config.SEED,
    use_gpu: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AttemptOutcome:
    """Run one plan end to end. `result.metrics` are holdout metrics;
    `result.attempt.cv` holds the training-split CV summary."""
    prepared = prepare_dataset(df, requirement, seed=random_state)
    outcome = run_attempt(
        plan,
        prepared,
        generate_shap=generate_shap,
        use_gpu=use_gpu,
        progress_callback=progress_callback,
    )
    t0 = time.perf_counter()
    holdout = holdout_evaluate(outcome.pipeline, prepared)
    outcome.result.diagnostics.append(
        BlockDiagnostics(block="holdout", duration_seconds=time.perf_counter() - t0, details=holdout)
    )
    outcome.result.metrics = holdout
    return outcome
