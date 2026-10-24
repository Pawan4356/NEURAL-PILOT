"""Hyperparameter optimisation (engine component 5).

Optuna on cross-validation over the training rows only. The objective is
the run's primary metric (audit G8), trials are bounded by a trial cap and
a timeout (audit G9), weak trials are pruned fold-by-fold, and a
cancellation event stops the study between trials.
"""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from ..schemas import ProblemType

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOResult:
    best_params: dict = field(default_factory=dict)
    best_value: float | None = None
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0
    stopped_reason: str | None = None
    duration_seconds: float = 0.0


def search_space(model_family: str, trial: optuna.Trial, problem_type: ProblemType) -> dict:
    """Search spaces keyed by *resolved* model_family registry name."""
    if model_family == "LinearModel":
        if problem_type == "classification":
            return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        return {"alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True)}
    if model_family == "DecisionTree":
        return {"max_depth": trial.suggest_int("max_depth", 2, 20)}
    if model_family == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
        }
    if model_family == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    if model_family == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    if model_family == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 200),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    return {}


def tune(
    make_pipeline: Callable[[dict], Pipeline],
    *,
    model_family: str,
    problem_type: ProblemType,
    sampler_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: np.ndarray | None,
    scorer_name: str,
    n_trials: int,
    timeout_s: float | None,
    seed: int,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> HPOResult:
    start = time.perf_counter()
    if sampler_name == "none" or n_trials <= 0:
        return HPOResult(stopped_reason="no tuning requested")

    scorer = get_scorer(scorer_name)
    folds = list(cv.split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = search_space(model_family, trial, problem_type)
        template = make_pipeline(params)
        scores: list[float] = []
        for step, (train_pos, valid_pos) in enumerate(folds):
            pipeline = clone(template)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X.iloc[train_pos], y.iloc[train_pos])
                scores.append(float(scorer(pipeline, X.iloc[valid_pos], y.iloc[valid_pos])))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = (
        optuna.samplers.RandomSampler(seed=seed)
        if sampler_name == "random"
        else optuna.samplers.TPESampler(seed=seed)
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress_callback:
            progress_callback(f"Hyperparameter tuning: trial {len(study.trials)}/{n_trials}")
        if cancel_event is not None and cancel_event.is_set():
            study.stop()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_s,
        catch=(Exception,),
        show_progress_bar=False,
        callbacks=[on_trial_end],
    )

    states = [t.state for t in study.trials]
    result = HPOResult(
        n_trials_completed=states.count(optuna.trial.TrialState.COMPLETE),
        n_trials_pruned=states.count(optuna.trial.TrialState.PRUNED),
        n_trials_failed=states.count(optuna.trial.TrialState.FAIL),
        duration_seconds=time.perf_counter() - start,
    )
    if result.n_trials_completed:
        result.best_params = dict(study.best_params)
        result.best_value = float(study.best_value)

    if cancel_event is not None and cancel_event.is_set():
        result.stopped_reason = "cancelled"
    elif not result.n_trials_completed:
        result.stopped_reason = "all trials failed; used default parameters"
    elif len(study.trials) < n_trials:
        result.stopped_reason = f"timeout after {len(study.trials)}/{n_trials} trials"
    else:
        result.stopped_reason = "completed"
    return result
