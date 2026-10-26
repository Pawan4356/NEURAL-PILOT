"""In-process caches for the heavy objects graph state only refers to.

- DataFrames, keyed by dataset fingerprint (reloaded from disk on a miss).
- `PreparedData` (profile, split, leakage scan), keyed by fingerprint +
  requirement + seed, rebuilt deterministically on a miss.
- The best fitted pipeline of each run, kept in memory and written to the
  run's model path, so a run resumed after a restart can still finalize.

Caches are bounded and thread-safe; several runs may execute at once.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .. import config
from ..engine import PreparedData, load_dataset, prepare_dataset

MAX_CACHED_DATASETS = 4

_lock = threading.Lock()
_frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
_prepared: OrderedDict[tuple, PreparedData] = OrderedDict()
_pipelines: dict[str, Pipeline] = {}


def _remember(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > MAX_CACHED_DATASETS:
        cache.popitem(last=False)


def remember_dataframe(fingerprint: str, df: pd.DataFrame) -> None:
    with _lock:
        _remember(_frames, fingerprint, df)


def dataframe(state: dict) -> pd.DataFrame:
    fingerprint = state["dataset_fingerprint"]
    with _lock:
        df = _frames.get(fingerprint)
        if df is not None:
            _frames.move_to_end(fingerprint)
            return df
    loaded = load_dataset(Path(state["dataset_path"]))
    if loaded.fingerprint != fingerprint:
        raise RuntimeError(
            f"Dataset file {state['dataset_path']} changed since the run started "
            f"(fingerprint {loaded.fingerprint} != {fingerprint})."
        )
    remember_dataframe(fingerprint, loaded.df)
    return loaded.df


def prepared_data(state: dict) -> PreparedData:
    requirement = state["requirement"]
    seed = state.get("seed", config.SEED)
    key = (state["dataset_fingerprint"], requirement.model_dump_json(), seed, config.LEAKAGE_ACTION)
    with _lock:
        prepared = _prepared.get(key)
        if prepared is not None:
            _prepared.move_to_end(key)
            return prepared
    prepared = prepare_dataset(dataframe(state), requirement, seed=seed)
    with _lock:
        _remember(_prepared, key, prepared)
    return prepared


def model_path(run_id: str) -> Path:
    return config.MODELS_DIR / f"{run_id}.joblib"


def keep_best_pipeline(run_id: str, pipeline: Pipeline) -> Path:
    """Hold the new best pipeline in memory and on disk."""
    path = model_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    with _lock:
        _pipelines[run_id] = pipeline
    return path


def best_pipeline(run_id: str) -> Pipeline:
    with _lock:
        pipeline = _pipelines.get(run_id)
    if pipeline is not None:
        return pipeline
    path = model_path(run_id)
    if not path.exists():
        raise RuntimeError(f"The best pipeline of run {run_id} is neither in memory nor at {path}.")
    pipeline = joblib.load(path)
    with _lock:
        _pipelines[run_id] = pipeline
    return pipeline


def release_run(run_id: str) -> None:
    """Drop a finished run's pipeline from memory (the file stays)."""
    with _lock:
        _pipelines.pop(run_id, None)
