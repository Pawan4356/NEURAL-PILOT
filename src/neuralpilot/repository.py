"""Experiment repository (§5). The implementation moved to `neuralpilot.memory`
in Phase 7 (store, similarity, migrations). This module re-exports the public
names so existing imports keep working."""

from .memory.migrations import BASE_SCHEMA as _BASE_SCHEMA
from .memory.migrations import SCHEMA_VERSION
from .memory.store import (
    EVIDENCE_OK,
    EVIDENCE_SUSPECT_LEAKAGE,
    ExperimentRecord,
    ExperimentRepository,
    SimilarExperiment,
    is_usable_evidence,
    plan_components_registered,
    retrieve_similar_experiments,
)

__all__ = [
    "EVIDENCE_OK",
    "EVIDENCE_SUSPECT_LEAKAGE",
    "SCHEMA_VERSION",
    "_BASE_SCHEMA",
    "ExperimentRecord",
    "ExperimentRepository",
    "SimilarExperiment",
    "is_usable_evidence",
    "plan_components_registered",
    "retrieve_similar_experiments",
]
