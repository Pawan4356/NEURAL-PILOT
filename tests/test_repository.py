import pandas as pd
import pytest

from agentic_automl.meta_features import extract_meta_features
from agentic_automl.repository import ExperimentRecord, ExperimentRepository


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
    results = repo.retrieve_similar_experiments(query_mf, k=2)

    assert len(results) == 2
    assert results[0].dataset_name == "mostly_numeric"


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
