import numpy as np
import pandas as pd
import pytest

from neuralpilot.profiling import derive_problem_type, leakage_scan, profile_columns


def _roles(df, target=None):
    return {p.name: p.role for p in profile_columns(df, target_column=target)}


def test_identifier_columns_are_detected_by_name_and_uniqueness():
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "CustomerID": rng.permutation(n) + 1000,
            "order_id": [f"ORD{i:05d}" for i in rng.permutation(n)],
            "Customer Id": [f"{v:x}" for v in rng.integers(1e9, 2e9, n)],
            "amount_paid": rng.normal(100, 10, n),  # ends in "id", but not an identifier
            "valid": rng.choice([0, 1], n),
            "y": rng.choice([0, 1], n),
        }
    )
    roles = _roles(df, "y")
    assert roles["CustomerID"] == "id"
    assert roles["order_id"] == "id"
    assert roles["Customer Id"] == "id"
    assert roles["amount_paid"] == "numeric"
    assert roles["valid"] == "categorical"


def test_float_typed_id_column_with_missing_value_is_still_an_id():
    # The churn CSV has a blank row, which turns CustomerID into float64.
    ids = pd.Series(np.arange(1, 301, dtype=float))
    ids.iloc[5] = np.nan
    df = pd.DataFrame({"CustomerID": ids, "y": [0, 1] * 150})
    assert _roles(df, "y")["CustomerID"] == "id"


def test_row_number_column_is_an_id_even_without_id_name():
    df = pd.DataFrame({"row": range(100), "x": np.random.default_rng(0).normal(size=100), "y": [0, 1] * 50})
    roles = _roles(df, "y")
    assert roles["row"] == "id"
    assert roles["x"] == "numeric"


def test_constant_datetime_and_text_roles():
    n = 60
    df = pd.DataFrame(
        {
            "const": ["same"] * n,
            "signup": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
            "review": [f"this product was great number {i} indeed" for i in range(n)],
            "city": ["Paris", "Rome", "Oslo"] * 20,
            "y": [0, 1] * 30,
        }
    )
    roles = _roles(df, "y")
    assert roles["const"] == "constant"
    assert roles["signup"] == "datetime"
    assert roles["review"] == "text"
    assert roles["city"] == "categorical"


def test_leakage_scan_flags_near_copy_of_target():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.choice([0, 1], n)
    df = pd.DataFrame(
        {
            "leak": y + rng.normal(scale=0.01, size=n),  # target + tiny noise
            "honest": y * 0.5 + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "y": y,
        }
    )
    flagged = {f.column for f in leakage_scan(df, "y", "classification")}
    assert flagged == {"leak"}


def test_leakage_scan_regression():
    rng = np.random.default_rng(2)
    n = 500
    y = rng.normal(size=n) * 10
    df = pd.DataFrame({"y_copy": y * 2 + 1, "x": rng.normal(size=n), "y": y})
    flagged = {f.column for f in leakage_scan(df, "y", "regression")}
    assert flagged == {"y_copy"}


def test_leakage_scan_multiclass_requires_margin_over_base_rate():
    rng = np.random.default_rng(3)
    n = 600
    y = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame({"label_code": pd.Series(y).map({"a": 1, "b": 2, "c": 3}), "x": rng.normal(size=n), "y": y})
    flagged = {f.column for f in leakage_scan(df, "y", "classification")}
    assert flagged == {"label_code"}


def test_churn_style_customer_id_is_excluded_by_the_engine():
    """Regression for audit finding 1: CustomerID ordered by churn status
    predicted Churn with 99% accuracy and was used as a feature."""
    from neuralpilot.engine import prepare_dataset
    from neuralpilot.schemas import ConstraintWeights, RequirementSpec

    rng = np.random.default_rng(4)
    n = 1000
    churn = np.sort(rng.choice([0, 1], n))[::-1]  # churners first, like the real file
    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "Support Calls": rng.integers(0, 10, n),
            "Total Spend": rng.normal(500, 100, n),
            "Churn": churn,
        }
    )
    requirement = RequirementSpec(
        target_column="Churn", target_confidence=1.0, problem_type="classification",
        constraint_weights=ConstraintWeights(), weight_confidence=1.0,
    )
    prepared = prepare_dataset(df, requirement)
    assert "CustomerID" in prepared.excluded_columns
    assert "CustomerID" not in prepared.X.columns


@pytest.mark.parametrize(
    "values, expected",
    [
        (["yes", "no"] * 50, "classification"),
        ([0, 1, 2] * 40, "classification"),
        (list(range(30)) * 30, "classification"),  # 30 integer labels, 3.3% of 900 rows
        (np.random.default_rng(0).normal(size=200).tolist(), "regression"),
        (list(range(1000)), "regression"),
    ],
)
def test_derive_problem_type(values, expected):
    assert derive_problem_type(pd.Series(values)) == expected
