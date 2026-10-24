"""Model factory (engine component 4).

Builds estimators from registry specs: seeds come from the registry
factories, GPU params are added only for families that support them, and
class weighting is applied where the estimator has a native option.
"""

from __future__ import annotations

import shutil
import subprocess

from ..registry import ComponentSpec
from ..schemas import ProblemType

# Native "balanced" class weighting per model family (classification only).
_CLASS_WEIGHT_PARAMS: dict[str, dict] = {
    "LinearModel": {"class_weight": "balanced"},
    "DecisionTree": {"class_weight": "balanced"},
    "RandomForest": {"class_weight": "balanced"},
    "ExtraTrees": {"class_weight": "balanced"},
    "HistGradientBoosting": {"class_weight": "balanced"},
    "CatBoost": {"auto_class_weights": "Balanced"},
}


def gpu_is_available() -> bool:
    """Return whether the host exposes an NVIDIA GPU through nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            timeout=5,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def resolve_gpu(use_gpu: bool | None) -> tuple[bool, str | None]:
    """Resolve the requested GPU mode and return (enabled, fallback reason)."""
    if use_gpu is False:
        return False, None
    if gpu_is_available():
        return True, None
    if use_gpu is True:
        return False, "GPU was requested but no usable NVIDIA GPU was detected"
    return False, None


def gpu_model_params(model_family: str, gpu_enabled: bool) -> dict:
    if not gpu_enabled:
        return {}
    if model_family == "XGBoost":
        return {"tree_method": "hist", "device": "cuda"}
    if model_family == "CatBoost":
        return {"task_type": "GPU", "devices": "0"}
    return {}


def class_weight_params(model_family: str, problem_type: ProblemType, balanced: bool) -> dict:
    if not balanced or problem_type != "classification":
        return {}
    return dict(_CLASS_WEIGHT_PARAMS.get(model_family, {}))


def supports_class_weight(model_family: str) -> bool:
    return model_family in _CLASS_WEIGHT_PARAMS


def build_estimator(
    spec: ComponentSpec,
    problem_type: ProblemType,
    params: dict,
    *,
    gpu_enabled: bool = False,
    balanced: bool = False,
):
    all_params = {
        **params,
        **gpu_model_params(spec.name, gpu_enabled),
        **class_weight_params(spec.name, problem_type, balanced),
    }
    return spec.build(problem_type, **all_params)
