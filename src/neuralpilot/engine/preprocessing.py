"""Preprocessing (engine component 2).

Builds the impute -> scale / impute -> encode ColumnTransformer. It is only
ever fitted inside a Pipeline, so every CV fold fits its own imputers,
scalers and encoders on that fold's training rows.

Planner-supplied `PreprocessingStep.params` are passed through only after
validation against a per-component whitelist; anything unknown or out of
range is dropped and recorded as a plan adjustment instead of being
silently ignored or crashing at fit time.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..registry import resolve_component
from ..schemas import PlanSpec

IMPUTATION = "median/most_frequent"  # fixed for now; part of the plan signature


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("must be true or false")
    return value


def _range_tuple(lo_min: float | None, hi_max: float | None) -> Callable[[Any], tuple[float, float]]:
    def check(value: Any) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("must be a pair [low, high]")
        lo, hi = float(value[0]), float(value[1])
        if lo >= hi or (lo_min is not None and lo < lo_min) or (hi_max is not None and hi > hi_max):
            raise ValueError("out of range")
        return (lo, hi)

    return check


def _min_frequency(value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError("must be a number")
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, float) and 0 < value < 1:
        return value
    raise ValueError("must be an integer >= 1 or a fraction in (0, 1)")


def _max_categories(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("must be an integer >= 2")
    return value


_PARAM_VALIDATORS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "OneHotEncoder": {"min_frequency": _min_frequency, "max_categories": _max_categories},
    "OrdinalEncoder": {},
    "StandardScaler": {"with_mean": _bool, "with_std": _bool},
    "MinMaxScaler": {"feature_range": _range_tuple(None, None), "clip": _bool},
    "RobustScaler": {
        "with_centering": _bool,
        "with_scaling": _bool,
        "quantile_range": _range_tuple(0.0, 100.0),
    },
}


def validate_params(component: str, params: dict) -> tuple[dict, list[str]]:
    """Keep only whitelisted, valid params. Returns (clean_params, notes)."""
    allowed = _PARAM_VALIDATORS.get(component, {})
    clean: dict = {}
    notes: list[str] = []
    for key, value in params.items():
        validator = allowed.get(key)
        if validator is None:
            notes.append(f"{component}: ignored unsupported parameter '{key}'")
            continue
        try:
            clean[key] = validator(value)
        except (TypeError, ValueError) as exc:
            notes.append(f"{component}: ignored invalid {key}={value!r} ({exc})")
    return clean, notes


def build_preprocessor(
    plan: PlanSpec,
    categorical_cols: list[str],
    numeric_cols: list[str],
    categorical_cardinality: int | None,
) -> tuple[ColumnTransformer, list[dict], list[str]]:
    """Returns (preprocessor, fallbacks, adjustments)."""
    fallbacks: list[dict] = []
    adjustments: list[str] = []
    encoder_step = next((s for s in plan.preprocessing if s.category == "encoder"), None)
    scaler_step = next((s for s in plan.preprocessing if s.category == "scaler"), None)

    encoder_spec, encoder_fb = resolve_component(
        "encoder",
        encoder_step.component if encoder_step else "OneHotEncoder",
        categorical_cardinality=categorical_cardinality,
    )
    if encoder_fb:
        fallbacks.append(encoder_fb.to_dict())
    scaler_spec, scaler_fb = resolve_component(
        "scaler", scaler_step.component if scaler_step else "StandardScaler"
    )
    if scaler_fb:
        fallbacks.append(scaler_fb.to_dict())

    # Params only apply to the component they were written for.
    encoder_params: dict = {}
    if encoder_step and encoder_step.params:
        if encoder_fb:
            adjustments.append(f"{encoder_step.component}: parameters dropped after fallback to {encoder_spec.name}")
        else:
            encoder_params, notes = validate_params(encoder_spec.name, encoder_step.params)
            adjustments.extend(notes)
    scaler_params: dict = {}
    if scaler_step and scaler_step.params:
        if scaler_fb:
            adjustments.append(f"{scaler_step.component}: parameters dropped after fallback to {scaler_spec.name}")
        else:
            scaler_params, notes = validate_params(scaler_spec.name, scaler_step.params)
            adjustments.extend(notes)

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_spec.build(**scaler_params))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder_spec.build(**encoder_params)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, fallbacks, adjustments


def estimate_encoded_width(
    encoder_name: str, numeric_cols: list[str], categorical_cardinalities: list[int]
) -> int:
    """Approximate post-preprocessing feature count (used by FE width guards)."""
    if encoder_name == "OneHotEncoder":
        return len(numeric_cols) + sum(categorical_cardinalities)
    return len(numeric_cols) + len(categorical_cardinalities)
