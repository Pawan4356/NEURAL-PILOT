from neuralpilot.registry import component_exists, get_interp_class, resolve_component


def test_known_component_resolves_without_fallback():
    spec, fallback = resolve_component("scaler", "StandardScaler")
    assert spec.name == "StandardScaler"
    assert fallback is None


def test_unknown_encoder_falls_back_by_cardinality_low():
    spec, fallback = resolve_component("encoder", "TargetEncoder", categorical_cardinality=5)
    assert spec.name == "OrdinalEncoder"
    assert fallback is not None
    assert fallback.to_dict() == {
        "category": "encoder",
        "requested": "TargetEncoder",
        "used": "OrdinalEncoder",
        "reason": "unsupported",
    }


def test_unknown_encoder_falls_back_by_cardinality_high():
    spec, fallback = resolve_component("encoder", "TargetEncoder", categorical_cardinality=50)
    assert spec.name == "OneHotEncoder"
    assert fallback.used == "OneHotEncoder"


def test_unknown_scaler_falls_back_to_standard_scaler():
    spec, fallback = resolve_component("scaler", "MysteryScaler")
    assert spec.name == "StandardScaler"
    assert fallback.reason == "unsupported"


def test_unknown_model_family_falls_back_to_random_forest():
    spec, fallback = resolve_component("model_family", "MysteryNet")
    assert spec.name == "RandomForest"
    assert fallback.used == "RandomForest"


def test_unknown_tuner_falls_back_to_optuna_random():
    spec, fallback = resolve_component("tuner", "MysteryTuner")
    assert spec.name == "OptunaRandom"


def test_component_exists():
    assert component_exists("model_family", "XGBoost")
    assert not component_exists("model_family", "NotARealModel")


def test_interp_class_lookup():
    assert get_interp_class("XGBoost") == "xgboost"
    assert get_interp_class("LinearModel") == "linear"
    assert get_interp_class("NotReal") is None


def test_model_family_build_produces_estimator_for_both_problem_types():
    spec, _ = resolve_component("model_family", "RandomForest")
    clf = spec.build("classification")
    reg = spec.build("regression")
    assert hasattr(clf, "fit") and hasattr(reg, "fit")
    assert type(clf).__name__ == "RandomForestClassifier"
    assert type(reg).__name__ == "RandomForestRegressor"
