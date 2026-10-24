"""Phase 4: registry v2 — aliases, semantic groups, alternatives, and the
new components (TargetEncoder, HistGradientBoosting, ExtraTrees)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from neuralpilot import registry
from neuralpilot.registry import REGISTRY, find_component, get_model_group, resolve_component


def test_aliases_resolve_to_the_canonical_component():
    assert find_component("model_family", "HistGB").name == "HistGradientBoosting"
    assert find_component("model_family", "LogisticRegression").name == "LinearModel"
    assert find_component("encoder", "ohe").name == "OneHotEncoder"  # case-insensitive
    spec, fallback = resolve_component("model_family", "GBM")
    assert spec.name == "GradientBoosting" and fallback is None


def test_every_alternative_is_a_registered_component_of_the_same_category():
    for category, table in REGISTRY.items():
        for spec in table.values():
            for alternative in spec.alternatives:
                assert alternative in table, f"{spec.name} lists unknown alternative {alternative}"
                assert alternative != spec.name


def test_model_groups_put_boosting_families_together():
    assert get_model_group("GradientBoosting") == get_model_group("HistGradientBoosting") == "gbdt"
    assert get_model_group("ExtraTrees") == get_model_group("RandomForest") == "bagging_tree"
    assert get_model_group("LinearModel") == "linear"


@pytest.mark.parametrize("name", sorted(REGISTRY["model_family"]))
@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_every_model_family_builds_and_fits_for_each_task(name, problem_type):
    spec = REGISTRY["model_family"][name]
    if problem_type not in spec.supports:
        pytest.skip(f"{name} does not support {problem_type}")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int) if problem_type == "classification" else X[:, 0] * 2 + rng.normal(size=120)
    model = spec.build(problem_type)
    model.fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_target_encoder_works_inside_a_pipeline_with_a_classification_target():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"city": rng.choice(["a", "b", "c", "d"], size=200)})
    y = (X["city"].isin(["a", "b"])).astype(int)
    encoder = REGISTRY["encoder"]["TargetEncoder"].build()
    preprocess = ColumnTransformer(
        [("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), ["city"])]
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", REGISTRY["model_family"]["LinearModel"].build("classification"))])
    pipeline.fit(X, y)
    assert pipeline.score(X, y) > 0.9


def test_new_families_are_known_to_the_interpretability_table():
    from neuralpilot import config
    from neuralpilot.registry import get_interp_class

    for family in ("HistGradientBoosting", "ExtraTrees"):
        assert get_interp_class(family) in config.INTERPRETABILITY_TABLE


def test_components_supporting_filters_by_task():
    assert "LinearModel" in registry.components_supporting("model_family", "regression")
    assert all("regression" in REGISTRY["model_family"][n].supports for n in registry.components_supporting("model_family", "regression"))
