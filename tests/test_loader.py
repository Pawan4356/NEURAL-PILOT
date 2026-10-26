from pathlib import Path

import pandas as pd
import pytest

from neuralpilot.engine.loader import DatasetLoadError, load_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def test_semicolon_file_with_decimal_commas_loads_all_columns():
    loaded = load_dataset(FIXTURES / "semicolon_decimal_comma.csv")
    assert list(loaded.df.columns) == ["school", "age", "studytime", "absences", "grade"]
    assert loaded.report["delimiter"] == ";"
    assert pd.api.types.is_float_dtype(loaded.df["grade"])
    assert loaded.df["grade"].between(5, 21).all()


def test_cp1252_file_is_decoded_instead_of_rejected():
    loaded = load_dataset(FIXTURES / "cp1252_accents.csv")
    assert loaded.report["encoding"] == "cp1252"
    assert loaded.df.shape == (15, 4)
    assert any("é" in name for name in loaded.df["track_name"])


def test_quoted_thousands_separators_become_numbers():
    loaded = load_dataset(FIXTURES / "thousands_separator.csv")
    assert pd.api.types.is_numeric_dtype(loaded.df["streams"])
    assert loaded.df["streams"].min() >= 1000
    assert "streams" in loaded.report["numeric_coerced_columns"]
    assert "song" not in loaded.report["numeric_coerced_columns"]


def test_junk_numeric_header_is_rejected_with_actionable_message():
    with pytest.raises(DatasetLoadError, match="header row"):
        load_dataset(FIXTURES / "junk_header.csv")


def test_single_column_file_is_rejected(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("value\n" + "\n".join(str(i) for i in range(20)))
    with pytest.raises(DatasetLoadError, match="single column"):
        load_dataset(path)


def test_too_few_rows_is_rejected(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(DatasetLoadError, match="at least 10"):
        load_dataset(path)


def test_row_and_column_limits(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("a,b,c\n" + "\n".join("1,2,3" for _ in range(20)))
    with pytest.raises(DatasetLoadError, match="columns"):
        load_dataset(path, max_cols=2)
    with pytest.raises(DatasetLoadError, match="rows"):
        load_dataset(path, max_rows=5)


def test_fingerprint_is_stable_and_content_sensitive(tmp_path):
    first = load_dataset(FIXTURES / "thousands_separator.csv").fingerprint
    again = load_dataset(FIXTURES / "thousands_separator.csv").fingerprint
    assert first == again

    changed = tmp_path / "changed.csv"
    changed.write_bytes((FIXTURES / "thousands_separator.csv").read_bytes() + b'song x,"1,000",0.5\n')
    assert load_dataset(changed).fingerprint != first


def test_plain_comma_csv_is_unchanged(tmp_path):
    df = pd.DataFrame({"x": range(20), "label": ["a", "b"] * 10, "score": [0.5] * 20})
    path = tmp_path / "plain.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.report["delimiter"] == ","
    assert loaded.report["numeric_coerced_columns"] == []
    pd.testing.assert_frame_equal(loaded.df, df, check_dtype=False)
