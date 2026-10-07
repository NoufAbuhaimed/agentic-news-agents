"""Graph wiring and the SQLite checkpointer that makes runs resumable."""

from __future__ import annotations

import sqlite3

import anthropic
import httpx
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from . import nodes
from .config import settings
from .state import DigestState

def _retryable(exc: Exception) -> bool:
    """Retry transient failures only; a 400/401/403 (bad request, auth, billing) won't fix itself."""
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (anthropic.APIConnectionError, httpx.TransportError))


# Network/API hiccups retry the node; a crash past that resumes from the last checkpoint next run.
_RETRY = RetryPolicy(max_attempts=3, initial_interval=10.0, backoff_factor=3.0, retry_on=_retryable)


_STATE_TYPES = ("ResearchThread", "Candidate", "Critique")


def build_graph():
    g = StateGraph(DigestState)
    g.add_node("planner", nodes.planner, retry_policy=_RETRY)
    g.add_node("researcher", nodes.researcher, retry_policy=_RETRY)
    g.add_node("fact_check", nodes.fact_check, retry_policy=_RETRY)
    g.add_node("select", nodes.select, retry_policy=_RETRY)
    g.add_node("writer", nodes.writer, retry_policy=_RETRY)
    g.add_node("critic", nodes.critic, retry_policy=_RETRY)
    g.add_node("send", nodes.send, retry_policy=_RETRY)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", nodes.fan_out, ["researcher"])
    g.add_edge("researcher", "fact_check")  # waits for all parallel agents
    g.add_conditional_edges("fact_check", nodes.after_fact_check, ["researcher", "select"])
    g.add_conditional_edges("select", nodes.after_select, {"writer": "writer", "end": END})
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", nodes.after_critic, {"writer": "writer", "send": "send"})
    g.add_edge("send", END)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Explicitly allow our own state types to be restored from checkpoints (nothing else).
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("digest.state", name) for name in _STATE_TYPES]
    )
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=serde)
    return g.compile(checkpointer=checkpointer)
