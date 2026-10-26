"""Reproducibility metadata recorded with every run: code version and the
versions of the libraries that determine execution results."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from importlib import metadata

from . import config

_TRACKED_PACKAGES = (
    "scikit-learn",
    "pandas",
    "numpy",
    "optuna",
    "xgboost",
    "catboost",
    "shap",
    "pydantic",
)


@lru_cache(maxsize=1)
def code_version() -> str:
    """Git SHA of the working tree, suffixed with '-dirty' if modified."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@lru_cache(maxsize=1)
def lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions
