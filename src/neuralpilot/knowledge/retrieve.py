"""Passage retrieval for the planner (Phase 8, Deliverable 7.5).

The query is a templated description of the dataset (task, class balance,
missing values, categorical share, skew), so the nearest passages describe
the situation rather than any particular column or value. Chunks are then
filtered by the task type they apply to and by whether every component they
name is still in the registry. A passage about a component that was removed
is therefore never shown as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..meta_features import DatasetMetaFeatures
from ..schemas import PlanSpec
from .store import KnowledgeStore


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    url: str
    section: str
    text: str
    trust: str
    distance: float

    def to_state(self) -> dict:
        """Plain dict for graph state, so the checkpointer can store it."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "section": self.section,
            "text": self.text,
            "trust": self.trust,
        }


def query_for(meta: DatasetMetaFeatures, problem_type: str | None, *, glass_box: bool = False) -> str:
    parts = [f"{problem_type or 'tabular'} task"]
    if problem_type == "classification" and meta.n_classes:
        parts.append(f"{meta.n_classes} classes")
        if meta.minority_class_ratio is not None and meta.minority_class_ratio < config.IMBALANCE_MINORITY_SHARE:
            parts.append("class imbalance")
    parts.append(f"{meta.categorical_ratio:.0%} categorical features")
    if meta.max_cardinality > 20:
        parts.append("high cardinality categories")
    if meta.missing_pct > 0:
        parts.append(f"{meta.missing_pct:.0f}% missing values")
    if meta.mean_abs_skew_numeric is not None and meta.mean_abs_skew_numeric > 1.0:
        parts.append("skewed numeric features")
    if meta.text_ratio > 0:
        parts.append("free text features")
    if meta.datetime_present:
        parts.append("date features")
    if glass_box:
        parts.append("interpretability required")
    return ", ".join(parts)


def retrieve(
    store: KnowledgeStore,
    query: str,
    *,
    problem_type: str | None,
    registry_components: set[str],
    k: int = config.RAG_MAX_PASSAGES,
    fetch: int = config.RAG_FETCH_N,
) -> list[Passage]:
    """Up to `k` passages for `query`, nearest first, that apply to `problem_type`
    and name only components the registry still has."""
    passages: list[Passage] = []
    for hit in store.query(query, n_results=fetch):
        if not _applies(hit.metadata, problem_type, registry_components):
            continue
        passages.append(
            Passage(
                id=hit.id,
                title=str(hit.metadata.get("title", "")),
                url=str(hit.metadata.get("url", "")),
                section=str(hit.metadata.get("section", "")),
                text=hit.text,
                trust=str(hit.metadata.get("trust", "")),
                distance=hit.distance,
            )
        )
        if len(passages) == k:
            break
    return passages


def check_citations(plan: PlanSpec, retrieved_ids: set[str]) -> tuple[PlanSpec, list[str]]:
    """Keep the plan's evidence ids that were retrieved for this run. An id that
    was not retrieved is not evidence, so it is dropped and named in a warning."""
    dropped = [i for i in plan.evidence_ids if i not in retrieved_ids]
    if not dropped:
        return plan, []
    kept = [i for i in plan.evidence_ids if i in retrieved_ids]
    warning = f"dropped evidence ids that were not retrieved for this run: {dropped}"
    return plan.model_copy(update={"evidence_ids": kept}), [warning]


def _applies(metadata: dict, problem_type: str | None, registry_components: set[str]) -> bool:
    tasks = _split(metadata.get("task_types"))
    if problem_type is not None and tasks and problem_type not in tasks:
        return False
    components = _split(metadata.get("component_ids"))
    return set(components) <= registry_components


def _split(value) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(",") if item]
