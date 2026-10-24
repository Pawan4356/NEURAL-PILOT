"""Feature engineering (engine component 3).

Not one of the §4.3 component categories (encoder/scaler/model_family/
tuner), so it does not go through the fallback decision tree — an
unrecognized step is skipped and logged, never a run failure.

Every parameter is bounded (audit G9): out-of-range values are clamped or
the step is dropped, and each change is returned as a plan adjustment.
"""

from __future__ import annotations

from typing import Callable

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures

from .. import config
from ..schemas import PlanSpec, ProblemType

KNOWN_STEPS = ("PolynomialFeatures", "SelectKBest", "PCA")


def _polynomial(params: dict, width: int, problem_type: ProblemType) -> tuple[object | None, list[str]]:
    notes: list[str] = []
    if width > config.POLYNOMIAL_MAX_INPUT_WIDTH:
        return None, [
            f"PolynomialFeatures dropped: ~{width} encoded features exceeds the "
            f"{config.POLYNOMIAL_MAX_INPUT_WIDTH}-feature limit"
        ]
    degree = params.get("degree", 2)
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        notes.append(f"PolynomialFeatures: invalid degree={degree!r}, used 2")
        degree = 2
    if degree > config.POLYNOMIAL_MAX_DEGREE:
        notes.append(f"PolynomialFeatures: degree {degree} clamped to {config.POLYNOMIAL_MAX_DEGREE}")
        degree = config.POLYNOMIAL_MAX_DEGREE
    interaction_only = params.get("interaction_only", False) is True
    return PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False), notes


def _select_k_best(params: dict, width: int, problem_type: ProblemType) -> tuple[object | None, list[str]]:
    notes: list[str] = []
    k = params.get("k", 10)
    if k != "all":
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            notes.append(f"SelectKBest: invalid k={k!r}, used 10")
            k = 10
        if k >= width:
            notes.append(f"SelectKBest: k={k} >= ~{width} available features, keeping all")
            k = "all"
    score_func = f_classif if problem_type == "classification" else f_regression
    return SelectKBest(score_func=score_func, k=k), notes


def _pca(params: dict, width: int, problem_type: ProblemType) -> tuple[object | None, list[str]]:
    notes: list[str] = []
    n = params.get("n_components", 0.95)
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        notes.append(f"PCA: invalid n_components={n!r}, used 0.95")
        n = 0.95
    elif isinstance(n, float) and not 0 < n <= 1:
        notes.append(f"PCA: n_components={n} outside (0, 1], used 0.95")
        n = 0.95
    elif isinstance(n, int) and n < 1:
        notes.append(f"PCA: n_components={n} < 1, used 0.95")
        n = 0.95
    elif isinstance(n, int) and n >= width:
        notes.append(f"PCA: n_components={n} >= ~{width} features, used 0.95 variance instead")
        n = 0.95
    if isinstance(n, float) and n == 1.0:
        n = 0.9999
    return PCA(n_components=n, random_state=config.SEED), notes


_FACTORIES: dict[str, Callable[[dict, int, ProblemType], tuple[object | None, list[str]]]] = {
    "PolynomialFeatures": _polynomial,
    "SelectKBest": _select_k_best,
    "PCA": _pca,
}


def build_fe_steps(
    plan: PlanSpec, problem_type: ProblemType, encoded_width: int
) -> tuple[list[tuple[str, object]], list[str], list[str]]:
    """Returns (steps, warnings, adjustments). `encoded_width` is the
    approximate feature count entering the first step."""
    steps: list[tuple[str, object]] = []
    warnings: list[str] = []
    adjustments: list[str] = []
    width = max(encoded_width, 1)
    for i, fe_step in enumerate(plan.feature_engineering):
        factory = _FACTORIES.get(fe_step.name)
        if factory is None:
            warnings.append(f"unknown feature_engineering step '{fe_step.name}' skipped")
            continue
        transformer, notes = factory(dict(fe_step.params), width, problem_type)
        adjustments.extend(notes)
        if transformer is None:
            continue
        steps.append((f"fe_{i}_{fe_step.name}", transformer))
        # Track width for later steps (approximate for PCA variance targets).
        if isinstance(transformer, PolynomialFeatures):
            d = transformer.degree
            width = width + width * (width + 1) // 2 if d == 2 else width
        elif isinstance(transformer, SelectKBest) and transformer.k != "all":
            width = int(transformer.k)
        elif isinstance(transformer, PCA) and isinstance(transformer.n_components, int):
            width = transformer.n_components
    return steps, warnings, adjustments
