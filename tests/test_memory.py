"""Phase 7: experiment memory v2 (Deliverable 7, tests in 10.1 'Memory' and 'HPO')."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from optuna.study import Study
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from neuralpilot.engine.hpo import search_space, tune, within_space
from neuralpilot.memory.migrations import BASE_SCHEMA, MIGRATIONS, SCHEMA_VERSION
from neuralpilot.memory.similarity import memory_vector, similarity
from neuralpilot.meta_features import DatasetMetaFeatures, extract_meta_features
from neuralpilot.repository import ExperimentRecord, ExperimentRepository
from neuralpilot.schemas import AttemptSummary


@pytest.fixture
def repo(tmp_path):
    r = ExperimentRepository(db_path=tmp_path / "memory.sqlite3")
    yield r
    r.close()


def _meta(n_rows=5_000, n_cols=10, categorical_ratio=0.2, missing_pct=1.0, **extra) -> dict:
    """A stored-style meta-feature dict with the fields the vector reads."""
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_classes": 2,
        "missing_pct": missing_pct,
        "categorical_ratio": categorical_ratio,
        "max_cardinality": 12,
        "minority_class_ratio": 0.4,
        "columns": [],
        **extra,
    }


def _stored(repo, name="churn_like", meta=None, **overrides) -> ExperimentRecord:
    fields = dict(
        dataset_name=name,
        problem_type="classification",
        target_column="y",
        meta_features=meta or _meta(),
        plan={"model_family": "GradientBoosting"},
        scores=[0.8],
        weakest_block_history=[],
        final_decision="accept",
        final_composite_score=0.8,
        final_report={"feasible": True, "model_family": "GradientBoosting"},
    )
    record = ExperimentRecord.new(**{**fields, **overrides})
    repo.save_run(record)
    return record


def _query(meta=None) -> DatasetMetaFeatures:
    return DatasetMetaFeatures(**(meta or _meta()))


def _attempt(signature, cv_mean=0.8, **overrides) -> AttemptSummary:
    fields = dict(
        signature=signature,
        iteration=1,
        model_family="GradientBoosting",
        preprocessing=["OneHotEncoder", "StandardScaler"],
        feature_engineering=[],
        tuning="OptunaTPE(10)",
        primary_metric="f1_weighted",
        cv_mean=cv_mean,
        cv_std=0.01,
        composite_score=0.7,
        best_params={"n_estimators": 120, "learning_rate": 0.1},
    )
    return AttemptSummary(**{**fields, **overrides})


# --- idempotency and attempt storage -------------------------------------------


def test_saving_a_run_twice_keeps_one_row(repo):
    record = _stored(repo)
    repo.save_run(record)
    assert len(repo.all_runs()) == 1


def test_saving_attempts_twice_keeps_one_row_per_signature(repo):
    record = _stored(repo)
    attempts = [_attempt("sig-a"), _attempt("sig-b", cv_mean=0.7)]
    repo.save_attempts(record.run_id, attempts)
    repo.save_attempts(record.run_id, attempts)
    rows = repo.attempts_for_run(record.run_id)
    assert [r["signature"] for r in rows] == ["sig-a", "sig-b"]


def test_attempts_round_trip_with_their_hyperparameters(repo):
    record = _stored(repo)
    repo.save_attempts(record.run_id, [_attempt("sig-a", feasible=False)])
    [row] = repo.attempts_for_run(record.run_id)
    assert row["best_params"] == {"n_estimators": 120, "learning_rate": 0.1}
    assert row["feasible"] is False


# --- evidence gating -----------------------------------------------------------


def test_failed_suspect_infeasible_and_unregistered_runs_are_not_evidence(repo):
    _stored(repo, "usable")
    _stored(repo, "failed", status="failed", error="boom", final_composite_score=None, final_report=None)
    _stored(repo, "suspect", evidence_quality="suspect_leakage")
    _stored(repo, "infeasible", final_report={"feasible": False, "model_family": "GradientBoosting"})
    _stored(repo, "retired", plan={"model_family": "NoSuchModelFamily"})

    names = [r.record.dataset_name for r in repo.retrieve_similar_experiments(_query(), k=10, min_similarity=0.0)]
    assert names == ["usable"]


def test_evidence_gating_does_not_hide_the_stored_runs(repo):
    _stored(repo, "suspect", evidence_quality="suspect_leakage")
    assert {r.dataset_name for r in repo.all_runs()} == {"suspect"}


def test_old_churn_runs_without_a_report_are_marked_suspect_by_the_migration(tmp_path):
    # A database as the audit found it: migrations 1 and 2 applied, churn runs stored without a report.
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(BASE_SCHEMA)
    for statements in MIGRATIONS[:2]:
        for statement in statements:
            conn.execute(statement)
    conn.execute("PRAGMA user_version = 2")
    meta = json.dumps(_meta())
    conn.executemany(
        "INSERT INTO runs (run_id, created_at, dataset_name, problem_type, target_column, meta_features_json,"
        " plan_json, scores_json, weakest_block_history_json, final_decision, final_composite_score)"
        " VALUES (?, '2026-01-01', ?, 'classification', 'Churn', ?, '{}', '[0.8]', '[]', 'stop', 0.8)",
        [("r-churn", "customer_churn", meta), ("r-other", "diabetes", meta)],
    )
    conn.commit()
    conn.close()

    repo = ExperimentRepository(db_path=db_path)
    try:
        assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        quality = {r.run_id: r.evidence_quality for r in repo.all_runs()}
        assert quality == {"r-churn": "suspect_leakage", "r-other": "ok"}
    finally:
        repo.close()


# --- warm start: the parameters must still fit the current search space --------


def test_within_space_keeps_only_values_the_search_can_express():
    space = search_space("GradientBoosting", "classification")
    params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 4, "bogus": 1}
    assert within_space(space, params) == {"n_estimators": 120, "learning_rate": 0.05}


def test_within_space_drops_out_of_range_and_non_integer_values():
    space = search_space("GradientBoosting", "classification")
    assert within_space(space, {"n_estimators": 9_999, "learning_rate": 0.5}) == {}
    assert within_space(space, {"n_estimators": 120.5}) == {}
    assert within_space(space, {"n_estimators": True}) == {}


def test_warm_start_is_the_first_trial_and_is_the_only_enqueued_one(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = (X["a"] > 0).astype(int)
    enqueued = []
    real_enqueue = Study.enqueue_trial

    def record_enqueue(self, params, **kwargs):
        enqueued.append(dict(params))
        return real_enqueue(self, params, **kwargs)

    monkeypatch.setattr(Study, "enqueue_trial", record_enqueue)
    tune(
        lambda params: Pipeline([("model", DecisionTreeClassifier(random_state=0, **params))]),
        model_family="DecisionTree",
        problem_type="classification",
        sampler_name="tpe",
        X=X,
        y=y,
        cv=StratifiedKFold(n_splits=3),
        groups=None,
        scorer_name="accuracy",
        n_trials=3,
        timeout_s=None,
        seed=0,
        warm_start_params={"max_depth": 4, "not_a_parameter": 7},
    )
    assert enqueued == [{"max_depth": 4}]


def test_no_warm_start_means_no_enqueued_trial(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120)})
    y = (X["a"] > 0).astype(int)
    enqueued = []
    monkeypatch.setattr(Study, "enqueue_trial", lambda self, params, **kw: enqueued.append(params))
    tune(
        lambda params: Pipeline([("model", DecisionTreeClassifier(random_state=0, **params))]),
        model_family="DecisionTree",
        problem_type="classification",
        sampler_name="tpe",
        X=X,
        y=y,
        cv=StratifiedKFold(n_splits=3),
        groups=None,
        scorer_name="accuracy",
        n_trials=2,
        timeout_s=None,
        seed=0,
    )
    assert enqueued == []


# --- meta-features v2 and the scaled vector ------------------------------------


def test_meta_features_v2_see_dates_free_text_and_skew():
    n = 200
    df = pd.DataFrame(
        {
            "when": pd.date_range("2020-01-01", periods=n, freq="D"),
            "note": [f"unique free text entry number {i}" for i in range(n)],
            "spend": np.exp(np.linspace(0, 5, n)),  # strongly right-skewed
            "y": [i % 2 for i in range(n)],
        }
    )
    meta = extract_meta_features(df, target_column="y")
    assert meta.datetime_present is True
    assert meta.text_ratio == pytest.approx(1 / 3)
    assert meta.mean_abs_skew_numeric is not None and meta.mean_abs_skew_numeric > 0.5


def test_legacy_records_read_the_new_dimensions_as_neutral():
    legacy = _meta()
    explicit = {**legacy, "text_ratio": 0.0, "datetime_present": False, "mean_abs_skew_numeric": None}
    assert memory_vector(legacy, "classification") == memory_vector(explicit, "classification")
    assert len(memory_vector(legacy, "classification")) == 10


def test_identical_meta_features_have_similarity_one_and_distance_lowers_it():
    base = memory_vector(_meta(), "classification")
    far = memory_vector(_meta(n_rows=50, missing_pct=60.0, categorical_ratio=0.9), "classification")
    assert similarity(base, base) == pytest.approx(1.0)
    assert similarity(base, far) < similarity(base, base)
