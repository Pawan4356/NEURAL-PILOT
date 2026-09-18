"""Component Registry + deterministic fallback (§4.3).

_XYZ -> Only for internal use, don't call it outside
__XYZ -> It helps prevent accidental access/overriding by subclasses (Python Name Mangling)
__XYZ__ -> Dunder/Special methods

Pure Python, no LLM involved. This is the foundation every other layer
calls: the Planning Agent names components by string, the Execution
Engine resolves those names to real objects through this registry, and
unsupported names fall back through a hardcoded decision tree rather than
similarity search.

Categories: encoder, scaler, model_family, tuner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

ProblemType = Literal["classification", "regression"]
Category = Literal["encoder", "scaler", "model_family", "tuner"]


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    category: Category
    build: Callable[..., Any] # A callable that can accept any arguments and can return any type of value.
    # Only meaningful for category == "model_family"; keys of
    # config.INTERPRETABILITY_TABLE (§4.4).
    interp_class: str | None = None
    supports: tuple[ProblemType, ...] = ("classification", "regression")


@dataclass
class FallbackRecord:
    category: str
    requested: str
    used: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "requested": self.requested,
            "used": self.used,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
_ENCODERS: dict[str, ComponentSpec] = {
    # Categories with a clear, logical rank
    "OrdinalEncoder": ComponentSpec(
        "OrdinalEncoder",
        "encoder",
        lambda **kw: OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, **kw
        ),
    ),
    # Categories with no inherent order
    "OneHotEncoder": ComponentSpec(
        "OneHotEncoder",
        "encoder",
        lambda **kw: OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kw),
    ),
}

# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------
_SCALERS: dict[str, ComponentSpec] = {
    # Removes the mean and scales data to have a standard deviation of 1 (mean = 0, variance = 1).
    "StandardScaler": ComponentSpec("StandardScaler", "scaler", lambda **kw: StandardScaler(**kw)),
    # Rescales every feature to a fixed range, usually between 0 and 1 (or -1 to 1 if negative values exist).
    "MinMaxScaler": ComponentSpec("MinMaxScaler", "scaler", lambda **kw: MinMaxScaler(**kw)),
    #  Uses the median and the Interquartile Range (IQR) instead of the mean and standard deviation.
    "RobustScaler": ComponentSpec("RobustScaler", "scaler", lambda **kw: RobustScaler(**kw)),
}

# ---------------------------------------------------------------------------
# Model families (classification / regression variants share a registry
# name; the build() factory picks the right sklearn class from problem_type)
# ---------------------------------------------------------------------------


def _linear_model(problem_type: ProblemType, **kw):
    if problem_type == "classification":
        kw.setdefault("random_state", 42)
        return LogisticRegression(max_iter=1000, **kw)
    if kw:
        kw.setdefault("random_state", 42)
        # Regression with Regularization (L2)
        # Solves the instability and overfitting of standard linear regression 
        # when independent variables are highly collinear.
        return Ridge(**kw)
    return LinearRegression()  # closed-form, already deterministic


def _tree_model(problem_type: ProblemType, **kw):
    kw.setdefault("random_state", 42)
    return (
        DecisionTreeClassifier(**kw)
        if problem_type == "classification"
        else DecisionTreeRegressor(**kw)
    )


def _random_forest(problem_type: ProblemType, **kw):
    kw.setdefault("random_state", 42)
    return (
        RandomForestClassifier(**kw)
        if problem_type == "classification"
        else RandomForestRegressor(**kw)
    )


def _gbm(problem_type: ProblemType, **kw):
    kw.setdefault("random_state", 42)
    return (
        GradientBoostingClassifier(**kw)
        if problem_type == "classification"
        else GradientBoostingRegressor(**kw)
    )


def _xgboost(problem_type: ProblemType, **kw):
    kw.setdefault("random_state", 42)
    if problem_type == "classification":
        return XGBClassifier(eval_metric="logloss", **kw)
    return XGBRegressor(**kw)


def _catboost(problem_type: ProblemType, **kw):
    kw.setdefault("verbose", False)
    kw.setdefault("random_state", 42)
    return CatBoostClassifier(**kw) if problem_type == "classification" else CatBoostRegressor(**kw)


_MODEL_FAMILIES: dict[str, ComponentSpec] = {
    "LinearModel": ComponentSpec("LinearModel", "model_family", _linear_model, interp_class="linear"),
    "DecisionTree": ComponentSpec("DecisionTree", "model_family", _tree_model, interp_class="tree"),
    "RandomForest": ComponentSpec(
        "RandomForest", "model_family", _random_forest, interp_class="random_forest"
    ),
    "GradientBoosting": ComponentSpec("GradientBoosting", "model_family", _gbm, interp_class="gbm"),
    "XGBoost": ComponentSpec("XGBoost", "model_family", _xgboost, interp_class="xgboost"),
    "CatBoost": ComponentSpec("CatBoost", "model_family", _catboost, interp_class="catboost"),
}

# ---------------------------------------------------------------------------
# Tuners — build() returns a config dict consumed by the Execution Engine's
# tuning step, not a live object (Optuna studies are created per-run).
# ---------------------------------------------------------------------------
_TUNERS: dict[str, ComponentSpec] = {
    # TPE stands for Tree-structured Parzen Estimator
    # Instead of treating every trial independently, TPE uses information from previous trials to decide 
    # which hyperparameter combinations are worth trying next.
    "OptunaTPE": ComponentSpec(
        "OptunaTPE", "tuner", lambda **kw: {"sampler": "tpe", "n_trials": kw.get("n_trials", 25)}
    ),
    # Tuning using random sampling
    "OptunaRandom": ComponentSpec(
        "OptunaRandom",
        "tuner",
        lambda **kw: {"sampler": "random", "n_trials": kw.get("n_trials", 25)},
    ),
    "NoTuning": ComponentSpec("NoTuning", "tuner", lambda **kw: {"sampler": "none", "n_trials": 0}),
}

REGISTRY: dict[Category, dict[str, ComponentSpec]] = {
    "encoder": _ENCODERS,
    "scaler": _SCALERS,
    "model_family": _MODEL_FAMILIES,
    "tuner": _TUNERS,
}

# Deterministic fallback targets (§4.3) — never a similarity search.
_FALLBACK_DEFAULTS: dict[Category, str] = {
    "encoder": "OneHotEncoder",  # overridden below when cardinality is known
    "scaler": "StandardScaler",
    "model_family": "RandomForest",
    "tuner": "OptunaRandom",
}


def component_exists(category: Category, name: str) -> bool:
    return name in REGISTRY.get(category, {})


def list_components(category: Category) -> list[str]:
    return sorted(REGISTRY.get(category, {}).keys())


def resolve_component(
    category: Category, name: str, *, categorical_cardinality: int | None = None
) -> tuple[ComponentSpec, FallbackRecord | None]:
    """Look up a component by name; fall back deterministically if unknown.

    Returns (spec, fallback_record) where fallback_record is None when the
    requested component existed as-is.
    """
    table = REGISTRY.get(category)
    if table is None:
        raise ValueError(f"Unknown component category: {category}")

    if name in table:
        return table[name], None

    if category == "encoder":
        fallback_name = (
            "OrdinalEncoder"
            if categorical_cardinality is not None and categorical_cardinality <= 10
            else "OneHotEncoder"
        )
    else:
        fallback_name = _FALLBACK_DEFAULTS[category]

    record = FallbackRecord(category=category, requested=name, used=fallback_name, reason="unsupported")
    return table[fallback_name], record


def get_interp_class(model_family_name: str) -> str | None:
    spec = _MODEL_FAMILIES.get(model_family_name)
    return spec.interp_class if spec else None
