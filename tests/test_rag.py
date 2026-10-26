"""Phase 8: documentation RAG (Deliverable 7). Offline: a local hashing embedding
stands in for the Chroma default model, so these tests never download anything."""

import hashlib
import textwrap
from pathlib import Path

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from neuralpilot import config
from neuralpilot.graph import nodes
from neuralpilot.knowledge import evaluate as evaluate_mod
from neuralpilot.knowledge.chunks import MAX_WORDS, OVERLAP_WORDS, Section, chunk_sections, content_hash, windows
from neuralpilot.knowledge.fetch import host_allowed, html_sections
from neuralpilot.knowledge.ingest import ingest
from neuralpilot.knowledge.registry_cards import CARD_SOURCE_TYPE, registry_cards
from neuralpilot.knowledge.retrieve import check_citations, query_for, retrieve
from neuralpilot.knowledge.sources import load_sources, registry_component_names
from neuralpilot.knowledge.store import KnowledgeStore, flatten_metadata, open_index
from neuralpilot.meta_features import DatasetMetaFeatures
from neuralpilot.schemas import PlanSpec, PreprocessingStep, TuningStrategy

REPO_ROOT = Path(__file__).resolve().parents[1]


class HashingEmbedding(EmbeddingFunction):
    """Bag of hashed words: texts that share words are close, which is all these tests need."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = [0.0] * 128
            for word in text.lower().split():
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore(tmp_path / "index", embedding_function=HashingEmbedding())


def _chunk(text, url, *, component_ids=(), task_types=("classification", "regression"), section="Overview"):
    metadata = {
        "source_type": "doc",
        "title": url,
        "url": url,
        "section": section,
        "component_ids": list(component_ids),
        "task_types": list(task_types),
        "trust": "official",
    }
    return chunk_sections([Section(heading=section, text=text)], metadata)[0]


# --- chunks ---------------------------------------------------------------------


def test_short_text_is_one_window_and_long_text_overlaps():
    assert windows("a few words") == ["a few words"]
    words = [f"w{i}" for i in range(MAX_WORDS * 2)]
    pieces = windows(" ".join(words))
    assert len(pieces) >= 2
    assert pieces[0].split()[-OVERLAP_WORDS:] == pieces[1].split()[:OVERLAP_WORDS]


def test_chunk_ids_are_content_hashes_so_reingesting_changes_nothing():
    first = _chunk("the same text", "https://example.org/a")
    again = _chunk("the same text", "https://example.org/a")
    assert first.id == again.id == content_hash("https://example.org/a", "Overview", "the same text")
    assert first.id != _chunk("other text", "https://example.org/a").id


# --- html ----------------------------------------------------------------------


def test_navigation_and_scripts_are_dropped_and_main_content_is_kept():
    html = textwrap.dedent(
        """
        <html><head><title>Page</title><script>var x = 1;</script></head>
        <body>
          <nav>Previous Next Site menu</nav>
          <div class="sphinxsidebar">Sidebar links</div>
          <div class="body" role="main">
            <h2>Scaling</h2><p>Standardization removes the mean.</p>
            <h2>Encoding</h2><p>One hot encoding creates binary columns.</p>
          </div>
          <footer>Copyright</footer>
        </body></html>
        """
    )
    sections = html_sections(html)
    text = " ".join(s.text for s in sections)
    assert "Standardization removes the mean." in text and "One hot encoding" in text
    assert "Sidebar" not in text and "Previous" not in text and "var x" not in text
    assert [s.heading for s in sections] == ["Scaling", "Encoding"]


def test_only_allow_listed_hosts_can_be_fetched():
    assert host_allowed("https://scikit-learn.org/stable/modules/preprocessing.html")
    assert not host_allowed("https://example.org/page")
    assert not host_allowed("https://scikit-learn.org.evil.example/page")


# --- store ----------------------------------------------------------------------


def test_upserting_the_same_chunks_twice_keeps_one_copy(store):
    chunks = [_chunk("alpha beta gamma", "https://example.org/a"), _chunk("delta epsilon", "https://example.org/b")]
    store.upsert(chunks)
    store.upsert(chunks)
    assert store.count() == 2


def test_nearest_chunk_is_returned_first(store):
    store.upsert(
        [
            _chunk("one hot encoding creates binary columns for categories", "https://example.org/enc"),
            _chunk("gradient boosting uses a learning rate and shrinkage", "https://example.org/gbm"),
        ]
    )
    hits = store.query("binary columns for each category", n_results=2)
    assert hits[0].metadata["url"] == "https://example.org/enc"
    assert store.query("anything", n_results=0) == []


def test_delete_by_source_type_removes_only_that_type(store):
    doc = _chunk("documentation text here", "https://example.org/a")
    card = _chunk("registry text here", "registry://model_family/X")
    card = type(card)(id=card.id, text=card.text, metadata={**card.metadata, "source_type": CARD_SOURCE_TYPE})
    store.upsert([doc, card])
    store.delete_source_type(CARD_SOURCE_TYPE)
    assert store.count() == 1


def test_list_metadata_is_stored_as_a_comma_string():
    assert flatten_metadata({"component_ids": ["B", "A"], "n": 2, "none": None}) == {"component_ids": "A,B", "n": 2}


def test_no_index_means_no_passages(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "absent")
    assert open_index() is None


def test_switching_rag_off_means_no_passages(tmp_path, monkeypatch):
    (tmp_path / "chroma").mkdir()
    monkeypatch.setattr(config, "RAG_ENABLED", False)
    assert open_index(tmp_path / "chroma") is None


# --- retrieval filters -----------------------------------------------------------


def test_passages_for_the_other_task_type_are_filtered_out(store):
    store.upsert(
        [
            _chunk("regression metric discussion text", "https://example.org/reg", task_types=("regression",)),
            _chunk("classification metric discussion text", "https://example.org/clf", task_types=("classification",)),
        ]
    )
    passages = retrieve(
        store, "metric discussion text", problem_type="classification", registry_components=set()
    )
    assert [p.url for p in passages] == ["https://example.org/clf"]


def test_passages_about_a_component_that_left_the_registry_are_never_shown(store):
    store.upsert(
        [
            _chunk("ghost model parameters text", "https://example.org/ghost", component_ids=("GhostModel",)),
            _chunk("live model parameters text", "https://example.org/live", component_ids=("LinearModel",)),
        ]
    )
    passages = retrieve(
        store, "model parameters text", problem_type="classification", registry_components={"LinearModel"}
    )
    assert [p.url for p in passages] == ["https://example.org/live"]


def test_retrieval_returns_at_most_k_passages(store):
    store.upsert([_chunk(f"shared words topic {i}", f"https://example.org/{i}") for i in range(8)])
    assert len(retrieve(store, "shared words topic", problem_type=None, registry_components=set(), k=3)) == 3


def test_query_describes_the_dataset_without_column_names():
    meta = DatasetMetaFeatures(
        n_rows=100,
        n_cols=5,
        n_classes=2,
        missing_pct=12.0,
        categorical_ratio=0.4,
        columns=[{"name": "secret_column_name"}],
        minority_class_ratio=0.02,
        mean_abs_skew_numeric=2.0,
    )
    text = query_for(meta, "classification", glass_box=True)
    assert "2 classes" in text and "class imbalance" in text and "interpretability required" in text
    assert "secret_column_name" not in text


# --- citations ------------------------------------------------------------------


def _plan(evidence):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family="LinearModel",
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
        evidence_ids=evidence,
    )


def test_citations_to_retrieved_passages_are_kept_and_others_dropped():
    plan, warnings = check_citations(_plan(["p1", "made-up"]), {"p1", "p2"})
    assert plan.evidence_ids == ["p1"]
    assert len(warnings) == 1 and "made-up" in warnings[0]


def test_a_plan_with_valid_citations_passes_unchanged():
    plan = _plan(["p1"])
    checked, warnings = check_citations(plan, {"p1"})
    assert checked is plan and warnings == []


def test_a_retrieval_failure_is_logged_and_the_run_continues(caplog):
    class BrokenStore:
        def query(self, text, n_results):
            raise RuntimeError("index unreadable")

    ctx = type("Ctx", (), {"knowledge": BrokenStore(), "llm": None})()
    state = {
        "run_id": "r1",
        "requirement": type("Req", (), {"problem_type": "classification", "constraints": []})(),
        "meta_features": DatasetMetaFeatures(
            n_rows=10, n_cols=3, n_classes=2, missing_pct=0.0, categorical_ratio=0.0, columns=[]
        ).to_dict(),
    }
    assert nodes._documentation(state, ctx) == []
    assert "continuing without it" in caplog.text


# --- registry cards -------------------------------------------------------------


def test_registry_cards_cover_every_component_and_are_deterministic():
    cards = registry_cards()
    names = {card.metadata["component_ids"][0] for card in cards}
    assert names == registry_component_names()
    assert [c.id for c in cards] == [c.id for c in registry_cards()]


def test_registry_card_names_its_task_types_and_interpretability():
    card = next(c for c in registry_cards() if c.metadata["component_ids"] == ["LinearModel"])
    assert "classification" in card.metadata["task_types"]
    assert "Interpretability class: linear" in card.text


# --- sources and ingestion ------------------------------------------------------


def _write_source(directory: Path, body: str, name="source.yaml"):
    (directory / name).write_text(textwrap.dedent(body), encoding="utf-8")


def test_a_source_on_a_host_outside_the_allow_list_is_rejected(tmp_path):
    _write_source(tmp_path, "library: x\npages:\n  - url: https://example.org/p\n    title: t\n")
    with pytest.raises(ValueError, match="allow-list"):
        load_sources(tmp_path)


def test_a_source_naming_an_unknown_component_is_rejected(tmp_path):
    _write_source(
        tmp_path,
        "library: x\npages:\n  - url: https://scikit-learn.org/p\n    title: t\n    component_ids: [NopeModel]\n",
    )
    with pytest.raises(ValueError, match="not in the registry"):
        load_sources(tmp_path)


def test_ingestion_writes_pages_and_cards_and_is_idempotent(tmp_path, store):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_source(
        sources,
        """
        library: scikit-learn
        pages:
          - url: https://scikit-learn.org/stable/modules/tree.html
            title: trees
            component_ids: [DecisionTree]
        """,
    )
    pages = {
        "https://scikit-learn.org/stable/modules/tree.html": "<body><h2>Trees</h2><p>Decision trees split on impurity.</p></body>"
    }

    first = ingest(store, sources, fetch=lambda url: pages[url])
    count = store.count()
    second = ingest(store, sources, fetch=lambda url: pages[url])

    assert first.pages == 1 and first.cards == len(registry_cards())
    assert store.count() == count == second.index_size
    assert first.failed == []


def test_an_unreachable_page_is_reported_and_the_others_are_still_ingested(tmp_path, store):
    sources = tmp_path / "sources"
    sources.mkdir()
    _write_source(
        sources,
        """
        library: scikit-learn
        pages:
          - url: https://scikit-learn.org/stable/modules/tree.html
            title: trees
          - url: https://scikit-learn.org/stable/modules/linear_model.html
            title: linear
        """,
    )

    def fetch(url):
        if "tree" in url:
            raise OSError("connection reset")
        return "<body><h2>Linear</h2><p>Logistic regression uses regularization.</p></body>"

    report = ingest(store, sources, fetch=fetch)
    assert report.pages == 1
    assert len(report.failed) == 1 and "connection reset" in report.failed[0]


# --- evaluation -----------------------------------------------------------------


def test_recall_and_mrr_on_a_small_fixture_index(store):
    store.upsert(
        [
            _chunk("ridge regression penalty shrinks coefficients", "https://example.org/ridge"),
            _chunk("histogram boosting bins continuous features", "https://example.org/hgb"),
            _chunk("pruning stops unpromising trials", "https://example.org/prune"),
        ]
    )
    queries = [
        evaluate_mod.LabelledQuery("ridge penalty coefficients", ("https://example.org/ridge",)),
        evaluate_mod.LabelledQuery("histogram bins features", ("https://example.org/hgb",)),
        evaluate_mod.LabelledQuery("pruning unpromising trials", ("https://example.org/prune",)),
    ]
    result = evaluate_mod.evaluate(store, queries)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.mrr == pytest.approx(1.0)
    assert result.misses == ()


def test_the_labelled_query_set_has_enough_queries_and_only_known_pages():
    queries = evaluate_mod.load_queries(REPO_ROOT / "tests" / "rag_eval" / "queries.yaml")
    assert len(queries) >= 30
    known = {page.url for page in load_sources(REPO_ROOT / "knowledge_sources")}
    labelled = {url for query in queries for url in query.expected_urls}
    assert labelled <= known


def test_the_checked_in_source_list_loads_and_names_only_registry_components():
    pages = load_sources(REPO_ROOT / "knowledge_sources")
    assert len(pages) >= 10
    for page in pages:
        assert set(page.component_ids) <= registry_component_names()


def test_the_planner_prompt_lists_passages_with_their_ids():
    from neuralpilot.agents.planning_agent import generate_plan
    from neuralpilot.schemas import ConstraintWeights, RequirementSpec

    captured = {}

    class Client:
        def complete_json(self, system, user, schema, **kwargs):
            captured["user"] = user
            return _plan([])

    requirement = RequirementSpec(
        target_column="y",
        target_confidence=0.9,
        problem_type="classification",
        constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
        weight_confidence=0.9,
    )
    meta = DatasetMetaFeatures(n_rows=10, n_cols=3, n_classes=2, missing_pct=0.0, categorical_ratio=0.0, columns=[])
    passages = [{"id": "abc123", "title": "Encoders", "section": "One hot", "text": "unknown categories", "url": "u", "trust": "official"}]
    generate_plan(requirement, meta, [], client=Client(), passages=passages)
    assert "id=abc123 (Encoders, section: One hot): unknown categories" in captured["user"]
