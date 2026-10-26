"""Assemble and compile the orchestration graph.

`build_graph()` wires the nodes from `nodes.py` with the routers from
`routing.py`. `compiled_graph()` returns the process-wide compiled graph
with a SQLite checkpointer (`config.GRAPH_DB_PATH`, thread_id = run_id),
which `interrupt()` needs and which keeps every run inspectable after a
restart.
"""

from __future__ import annotations

import functools
import inspect
import logging
import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from .. import config, schemas
from . import nodes, routing
from .context import RunContext
from .state import NPState

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_compiled: dict[Path, object] = {}


def guarded(name: str, fn):
    """Run a node; turn an unexpected exception into a recorded error so the
    routers send the run to `fail`. Interrupts (and other LangGraph control
    signals) pass through untouched."""

    @functools.wraps(fn)
    def node(state: NPState, runtime: Runtime[RunContext]) -> dict:
        try:
            return fn(state, runtime)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            logger.exception("run=%s node=%s failed", state.get("run_id"), name)
            if name == "fail":  # never mask the original error
                return {"status": "failed"}
            return {"error": str(exc) or type(exc).__name__, "failed_node": name}

    return node


def build_graph() -> StateGraph:
    graph = StateGraph(NPState, context_schema=RunContext)
    for name in (
        "understand",
        "clarify",
        "prepare",
        "retrieve_knowledge",
        "start_iteration",
        "plan",
        "resolve_plan",
        "execute",
        "reflect",
        "decide",
        "finalize",
        "persist",
        "fail",
    ):
        graph.add_node(name, guarded(name, getattr(nodes, name)))

    graph.add_edge(START, "understand")
    graph.add_conditional_edges("understand", routing.after_understand, ["clarify", "prepare", "fail"])
    graph.add_conditional_edges("clarify", routing.after_clarify, ["clarify", "prepare", "fail"])
    graph.add_conditional_edges("prepare", routing.after("retrieve_knowledge"), ["retrieve_knowledge", "fail"])
    graph.add_conditional_edges("retrieve_knowledge", routing.after("start_iteration"), ["start_iteration", "fail"])
    graph.add_conditional_edges("start_iteration", routing.after_start_iteration, ["plan", "finalize", "fail"])
    graph.add_conditional_edges("plan", routing.after("resolve_plan"), ["resolve_plan", "fail"])
    graph.add_conditional_edges("resolve_plan", routing.after_resolve_plan, ["execute", "plan", "finalize", "fail"])
    graph.add_conditional_edges("execute", routing.after("reflect"), ["reflect", "fail"])
    graph.add_conditional_edges("reflect", routing.after("decide"), ["decide", "fail"])
    graph.add_conditional_edges("decide", routing.after_decide, ["start_iteration", "finalize", "fail"])
    graph.add_conditional_edges("finalize", routing.after("persist"), ["persist", "fail"])
    graph.add_conditional_edges("persist", routing.after(END), [END, "fail"])
    graph.add_edge("fail", END)
    return graph


def checkpoint_serializer() -> JsonPlusSerializer:
    """Only NeuralPilot's own schema classes may be rebuilt from a checkpoint."""
    types = [
        obj
        for _, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=types)


def open_checkpointer(db_path: Path) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Runs execute in background threads; SqliteSaver serialises access itself.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn, serde=checkpoint_serializer())


def compiled_graph(db_path: Path | None = None):
    """The compiled graph bound to the checkpoint database at `db_path`
    (default `config.GRAPH_DB_PATH`), built once per path."""
    path = Path(db_path or config.GRAPH_DB_PATH)
    with _lock:
        graph = _compiled.get(path)
        if graph is None:
            graph = build_graph().compile(checkpointer=open_checkpointer(path))
            _compiled[path] = graph
        return graph
