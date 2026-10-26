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
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from ..schemas import ProblemType
from .progress import ProgressCallback

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


def search_space(model_family: str, problem_type: ProblemType) -> dict[str, BaseDistribution]:
    """Search spaces keyed by *resolved* model_family registry name.

    Declared as distributions, so a warm-start value can be checked against
    the same bounds the trials sample from (Phase 7)."""
    if model_family == "LinearModel":
        if problem_type == "classification":
            return {"C": FloatDistribution(0.01, 10.0, log=True)}
        return {"alpha": FloatDistribution(0.01, 10.0, log=True)}
    if model_family == "DecisionTree":
        return {"max_depth": IntDistribution(2, 20)}
    if model_family == "RandomForest":
        return {"n_estimators": IntDistribution(50, 300), "max_depth": IntDistribution(2, 20)}
    if model_family == "GradientBoosting":
        return {"n_estimators": IntDistribution(50, 200), "learning_rate": FloatDistribution(0.01, 0.3, log=True)}
    if model_family == "ExtraTrees":
        return {"n_estimators": IntDistribution(50, 300), "max_depth": IntDistribution(2, 20)}
    if model_family == "HistGradientBoosting":
        return {
            "max_iter": IntDistribution(50, 300),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "max_leaf_nodes": IntDistribution(8, 64),
        }
    if model_family == "XGBoost":
        return {
            "n_estimators": IntDistribution(50, 300),
            "max_depth": IntDistribution(2, 10),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
        }
    if model_family == "CatBoost":
        return {
            "iterations": IntDistribution(50, 200),
            "depth": IntDistribution(3, 8),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
        }
    return {}


def suggest_params(trial: optuna.Trial, space: dict[str, BaseDistribution]) -> dict:
    params: dict = {}
    for name, dist in space.items():
        if isinstance(dist, IntDistribution):
            params[name] = trial.suggest_int(name, dist.low, dist.high, log=dist.log)
        else:
            params[name] = trial.suggest_float(name, dist.low, dist.high, log=dist.log)
    return params


def within_space(space: dict[str, BaseDistribution], params: dict) -> dict:
    """The part of `params` that the current space can express: names that are
    still tunable, with values inside their bounds (and integral for integer
    parameters). Anything else is dropped rather than clipped, because clipping
    would silently test a configuration that was never recorded."""
    kept: dict = {}
    for name, value in params.items():
        dist = space.get(name)
        if dist is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not dist.low <= value <= dist.high:
            continue
        if isinstance(dist, IntDistribution):
            if not float(value).is_integer():
                continue
            kept[name] = int(value)
        else:
            kept[name] = float(value)
    return kept


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
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    warm_start_params: dict | None = None,
    device: str | None = None,
) -> HPOResult:
    start = time.perf_counter()
    if sampler_name == "none" or n_trials <= 0:
        return HPOResult(stopped_reason="no tuning requested")

    scorer = get_scorer(scorer_name)
    folds = list(cv.split(X, y, groups))
    space = search_space(model_family, problem_type)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)
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
    # A stored configuration is tried first, so the search starts from what
    # worked before. It counts as one of the trials, not an extra one.
    warm_start = within_space(space, warm_start_params or {})
    if warm_start:
        study.enqueue_trial(warm_start)

    def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress_callback:
            # One advancing line, not one line per trial: the UI draws the bar.
            progress_callback(
                "Tuning hyperparameters", progress=(len(study.trials), n_trials), device=device
            )
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
