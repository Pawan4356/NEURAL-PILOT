import json

import pandas as pd
import pytest

from neuralpilot.meta_features import extract_meta_features
from neuralpilot.repository import ExperimentRecord, ExperimentRepository


@pytest.fixture
def repo(tmp_path):
    r = ExperimentRepository(db_path=tmp_path / "test.sqlite3")
    yield r
    r.close()


def _make_df(n_rows, n_cols_extra=2):
    data = {f"x{i}": range(n_rows) for i in range(n_cols_extra)}
    data["y"] = [i % 2 for i in range(n_rows)]
    return pd.DataFrame(data)


def _record_for(df, dataset_name, target="y", **overrides):
    mf = extract_meta_features(df, target_column=target)
    defaults = dict(
        dataset_name=dataset_name,
        problem_type="classification",
        target_column=target,
        meta_features=mf.to_dict(),
        plan={"model_family": "RandomForest"},
        scores=[0.5, 0.6],
        weakest_block_history=["model_family"],
        final_decision="accept",
        final_composite_score=0.6,
    )
    defaults.update(overrides)
    return ExperimentRecord.new(**defaults)


def test_save_and_list_run(repo):
    df = _make_df(100)
    record = _record_for(df, "small_dataset")
    repo.save_run(record)

    runs = repo.all_runs()
    assert len(runs) == 1
    assert runs[0].run_id == record.run_id
    assert runs[0].dataset_name == "small_dataset"
    assert runs[0].scores == [0.5, 0.6]


def test_retrieve_similar_experiments_ranks_by_similarity(repo):
    # Same n_rows/n_cols/n_classes across all three so the only variation is
    # categorical_ratio, isolating the similarity comparison to that axis
    # (raw-vector cosine similarity is dominated by whichever dimension has
    # the largest magnitude, which is n_rows here, so we hold it constant).
    mostly_numeric_df = _make_df(200, n_cols_extra=8)  # 0/8 categorical
    mostly_categorical_df = pd.DataFrame(
        {**{f"c{i}": ["a", "b"] * 100 for i in range(7)}, "y": [0, 1] * 100}
    )  # 7/8 categorical
    query_df = pd.DataFrame(
        {**{f"c{i}": ["a", "b"] * 100 for i in range(1)}, **{f"x{i}": range(200) for i in range(7)}, "y": [0, 1] * 100}
    )  # 1/8 categorical — closer to mostly_numeric

    repo.save_run(_record_for(mostly_numeric_df, "mostly_numeric"))
    repo.save_run(_record_for(mostly_categorical_df, "mostly_categorical"))

    query_mf = extract_meta_features(query_df, target_column="y")
    results = repo.retrieve_similar_experiments(query_mf, k=2, min_similarity=0.0)

    assert len(results) == 2
    assert results[0].record.dataset_name == "mostly_numeric"
    assert results[0].similarity > results[1].similarity


def test_retrieve_respects_k(repo):
    df = _make_df(100)
    for i in range(5):
        repo.save_run(_record_for(df, f"run_{i}"))

    query_mf = extract_meta_features(df, target_column="y")
    results = repo.retrieve_similar_experiments(query_mf, k=3)
    assert len(results) == 3


def test_retrieve_empty_repository_returns_empty(repo):
    df = _make_df(100)
    query_mf = extract_meta_features(df, target_column="y")
    assert repo.retrieve_similar_experiments(query_mf, k=3) == []


# --- Audit G4: retrieval must discriminate between datasets ---------------
#
# Meta-features shaped like the audit's persisted runs (customer churn:
# 440k rows; diabetes: 768 rows; diabetes_risk: ~5k rows). With the old raw
# cosine similarity every pair scored >= 0.99993 because n_rows dominated.

def _meta(n_rows, n_cols, n_classes, missing_pct, categorical_ratio, max_cardinality, minority):
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_classes": n_classes,
        "missing_pct": missing_pct,
        "categorical_ratio": categorical_ratio,
        "columns": [],
        "max_cardinality": max_cardinality,
        "minority_class_ratio": minority,
        "target_skew": None,
    }


CHURN = _meta(440_832, 12, 2, 0.0, 0.42, 10, 0.43)
DIABETES = _meta(768, 9, 2, 0.0, 0.22, 17, 0.35)
DIABETES_RISK = _meta(5_000, 10, 2, 0.0, 0.30, 4, 0.30)


def _stored(repo, name, meta, problem_type="classification", **overrides):
    fields = dict(
        dataset_name=name,
        problem_type=problem_type,
        target_column="y",
        meta_features=meta,
        plan={"model_family": "GradientBoosting"},
        scores=[0.8],
        weakest_block_history=[],
        final_decision="accept",
        final_composite_score=0.8,
    )
    record = ExperimentRecord.new(**{**fields, **overrides})
    repo.save_run(record)
    return record


def _query(meta):
    from neuralpilot.meta_features import DatasetMetaFeatures

    return DatasetMetaFeatures(**meta)


def test_diabetes_nearest_neighbour_is_not_churn(repo):
    _stored(repo, "customer_churn", CHURN)
    _stored(repo, "diabetes_risk", DIABETES_RISK)

    results = repo.retrieve_similar_experiments(
        _query(DIABETES), k=2, problem_type="classification", min_similarity=0.0
    )
    assert results[0].record.dataset_name == "diabetes_risk"
    assert results[1].record.dataset_name == "customer_churn"
    assert results[0].similarity - results[1].similarity > 0.05


def test_regression_query_never_returns_classification_runs(repo):
    _stored(repo, "churn", CHURN, problem_type="classification")
    results = repo.retrieve_similar_experiments(
        _query(CHURN), k=3, problem_type="regression", min_similarity=0.0
    )
    assert results == []


def test_dissimilar_runs_are_dropped_below_threshold(repo):
    _stored(repo, "customer_churn", CHURN)
    tiny = _meta(50, 40, 7, 30.0, 0.9, 5_000, 0.02)
    assert repo.retrieve_similar_experiments(_query(tiny), k=3, min_similarity=0.5) == []


def test_failed_runs_are_stored_but_not_retrieved(repo):
    _stored(repo, "churn_ok", CHURN)
    _stored(repo, "churn_failed", CHURN, status="failed", error="boom", final_composite_score=None)

    assert {r.status for r in repo.all_runs()} == {"finalized", "failed"}
    results = repo.retrieve_similar_experiments(_query(CHURN), k=3, min_similarity=0.0)
    assert [r.record.dataset_name for r in results] == ["churn_ok"]


def test_new_fields_round_trip(repo):
    record = _stored(
        repo,
        "churn",
        CHURN,
        iteration_logs=[{"iteration": 1, "decision": "stop", "decision_forced": True}],
        final_metrics={"holdout": {"f1": 0.9}},
        model_path="/tmp/model.joblib",
        dataset_fingerprint="abc",
        code_version="deadbeef",
        lib_versions={"pandas": "3"},
    )
    loaded = repo.get_run(record.run_id)
    assert loaded.iteration_logs[0]["decision_forced"] is True
    assert loaded.final_metrics == {"holdout": {"f1": 0.9}}
    assert loaded.dataset_fingerprint == "abc"
    assert loaded.lib_versions == {"pandas": "3"}


def test_old_database_is_migrated_in_place(tmp_path):
    import sqlite3

    from neuralpilot.repository import SCHEMA_VERSION, _BASE_SCHEMA

    db_path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(_BASE_SCHEMA)
    conn.execute(
        "INSERT INTO runs VALUES ('r1', '2026-01-01', 'old', 'classification', 'y', ?, '{}', '[0.5]', '[]', 'stop', 0.5)",
        (json.dumps(CHURN),),
    )
    conn.commit()
    conn.close()

    repo = ExperimentRepository(db_path=db_path)
    try:
        assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        [old] = repo.all_runs()
        assert old.status == "finalized" and old.iteration_logs == []
        assert repo.retrieve_similar_experiments(_query(CHURN), k=1)[0].record.run_id == "r1"
    finally:
        repo.close()


def test_mlflow_run_id_and_unsupported_recommendations_round_trip(repo):
    from neuralpilot.schemas import UnsupportedRecommendation

    record = ExperimentRecord.new(
        run_id="run-1",
        dataset_name="d.csv",
        problem_type="classification",
        target_column="y",
        meta_features={},
        plan={},
        scores=[],
        weakest_block_history=[],
        final_decision="accept",
        final_composite_score=0.5,
        status="finalized",
        mlflow_run_id="abc123",
    )
    repo.save_run(record)
    assert repo.get_run("run-1").mlflow_run_id == "abc123"

    rows = [
        UnsupportedRecommendation(
            iteration=1,
            requested="XYZEncoder",
            category="encoder",
            source="planner",
            reason="not_in_registry",
            alternative_used="OneHotEncoder",
            semantic_change=False,
            context={"n_rows": 10},
        )
    ]
    repo.save_unsupported("run-1", rows)
    repo.save_unsupported("run-1", rows)  # saving again replaces, it does not duplicate
    [stored] = repo.unsupported_for_run("run-1")
    assert stored["requested"] == "XYZEncoder"
    assert stored["alternative_used"] == "OneHotEncoder"
    assert stored["semantic_change"] is False
    assert stored["context"] == {"n_rows": 10}
