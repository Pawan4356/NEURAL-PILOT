"""Plan Verification (§4.1) — deterministic, rule-based, runs before every execution.

All hardcoded checks must pass or the plan is rejected and sent back to the
Planning Agent with the failure reason. Component-existence failures are
*not* rejections: they resolve through the registry's deterministic
fallback (§4.3) and are logged, never blocking the run.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .meta_features import is_categorical_column
from .registry import resolve_component
from .schemas import PlanSpec, RequirementSpec, VerificationResult


def verify_plan(
    plan: PlanSpec, requirement: RequirementSpec, df: pd.DataFrame
) -> VerificationResult:
    failures: list[str] = []
    warnings: list[str] = []
    fallbacks: list[dict] = []

    target = requirement.target_column
    n_rows, n_cols = df.shape

    # --- target_column exists in dataset ---
    if target not in df.columns:
        failures.append(f"target_column '{target}' does not exist in dataset")
        # Nothing downstream about the target is checkable if it's missing.
        return VerificationResult(passed=False, failures=failures, warnings=warnings, fallbacks=fallbacks)

    target_series = df[target]

    # --- missing-value % on target column — block if >80% ---
    target_missing_pct = float(target_series.isna().mean() * 100)
    if target_missing_pct > config.TARGET_MISSING_BLOCK_PCT:
        failures.append(
            f"target_column '{target}' has {target_missing_pct:.1f}% missing values "
            f"(> {config.TARGET_MISSING_BLOCK_PCT}% block threshold)"
        )

    non_null_target = target_series.dropna()

    # --- target cardinality / type checks ---
    if requirement.problem_type == "classification":
        n_unique = int(non_null_target.nunique())
        if n_unique < 2:
            failures.append(
                f"target_column '{target}' has {n_unique} unique value(s); "
                "classification requires >= 2"
            )
        else:
            counts = non_null_target.value_counts()
            if len(counts) >= 2 and counts.min() > 0:
                imbalance_ratio = float(counts.max() / counts.min())
                if imbalance_ratio > config.CLASS_IMBALANCE_WARN_RATIO:
                    warnings.append(
                        f"class imbalance ratio {imbalance_ratio:.1f}:1 exceeds "
                        f"{config.CLASS_IMBALANCE_WARN_RATIO:.0f}:1 (warning only)"
                    )
    else:  # regression
        if not pd.api.types.is_numeric_dtype(target_series):
            failures.append(
                f"target_column '{target}' is not numeric; regression requires a numeric target"
            )

    # --- n_features > n_rows / 10 -> warning, not blocking ---
    n_features = n_cols - 1  # excluding target
    if n_rows > 0 and n_features > n_rows / 10:
        warnings.append(
            f"n_features ({n_features}) > n_rows/10 ({n_rows / 10:.1f}); "
            "high risk of overfitting (warning only)"
        )

    # --- component existence -> deterministic fallback, never blocks ---
    categorical_cardinality = None
    categorical_cols = [c for c in df.columns if c != target and is_categorical_column(df[c])]
    if categorical_cols:
        categorical_cardinality = int(max(df[c].nunique(dropna=True) for c in categorical_cols))

    for step in plan.preprocessing:
        _, fallback = resolve_component(
            step.category, step.component, categorical_cardinality=categorical_cardinality
        )
        if fallback:
            fallbacks.append(fallback.to_dict())

    _, model_fallback = resolve_component("model_family", plan.model_family)
    if model_fallback:
        fallbacks.append(model_fallback.to_dict())

    _, tuner_fallback = resolve_component("tuner", plan.tuning.component)
    if tuner_fallback:
        fallbacks.append(tuner_fallback.to_dict())

    passed = len(failures) == 0
    return VerificationResult(passed=passed, failures=failures, warnings=warnings, fallbacks=fallbacks)
