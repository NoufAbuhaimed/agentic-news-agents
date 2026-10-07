"""The main LangGraph graph: which steps (nodes) exist and how they connect (edges).

This file is the "map" of the whole system. Every `add_node` is one box in the architecture
diagram, every `add_edge` is a fixed arrow, and every `add_conditional_edges` is an arrow whose
destination is decided at run time by a small routing function.

    START → planner ─(Send ×4, parallel)→ researcher → fact_check ─┬─(retry weak topics)→ researcher
                                                                    └─→ select ─┬─(nothing new)→ END
                                                                                └─→ writer ⇄ critic → send → END

Non-sequential patterns used (what makes this more than a chain):
  * planner decomposes the query into topics
  * parallel branches: one research agent per topic, run at the same time (LangGraph `Send`)
  * fact-check retry loop back to the researcher
  * writer ⇄ critic quality loop
  * conditional routing (e.g. stop early when there is no new news)

It also sets up the checkpointer, which saves the shared state after every node so that a run
that crashes can resume from where it stopped instead of starting (and paying) again.
"""

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
    # Anthropic API errors: retry timeouts (408), conflicts (409), rate limits (429) and server errors (5xx).
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    # Plain HTTP errors (e.g. Signal API): retry rate limits and server errors.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    # Network problems (connection refused, DNS, timeouts) are worth retrying too.
    return isinstance(exc, (anthropic.APIConnectionError, httpx.TransportError))


# Network/API hiccups retry the node; a crash past that resumes from the last checkpoint next run.
# Up to 3 attempts, waiting 10s, then 30s (backoff_factor 3) between them.
_RETRY = RetryPolicy(max_attempts=3, initial_interval=10.0, backoff_factor=3.0, retry_on=_retryable)


# Our own Pydantic classes stored inside the state. The checkpointer only restores types it is
# explicitly told about (a safety feature against loading arbitrary objects from disk).
_STATE_TYPES = ("ResearchThread", "Candidate", "Critique")


def build_graph():
    # A StateGraph: every node receives the shared state (DigestState) and returns the keys it changes.
    g = StateGraph(DigestState)

    # --- Nodes (the boxes). Each one is a plain Python function in nodes.py. ---------------------
    g.add_node("planner", nodes.planner, retry_policy=_RETRY)        # splits the brief into 4 topics
    g.add_node("researcher", nodes.researcher, retry_policy=_RETRY)  # one create_agent research agent per topic
    g.add_node("fact_check", nodes.fact_check, retry_policy=_RETRY)  # checks findings against their sources
    g.add_node("select", nodes.select, retry_policy=_RETRY)          # dedupe + choose the best items
    g.add_node("writer", nodes.writer, retry_policy=_RETRY)          # writes the Signal message (Claude Sonnet)
    g.add_node("critic", nodes.critic, retry_policy=_RETRY)          # reviews the message
    g.add_node("send", nodes.send, retry_policy=_RETRY)              # posts to Signal + updates memory

    # --- Edges (the arrows). ------------------------------------------------------------------------
    g.add_edge(START, "planner")
    # Fan-out: fan_out() returns one Send per topic, so `researcher` runs ×4 in parallel (map step).
    g.add_conditional_edges("planner", nodes.fan_out, ["researcher"])
    g.add_edge("researcher", "fact_check")  # waits for all parallel agents (the "reduce" step)
    # Retry loop: after_fact_check() either sends weak topics back to `researcher`, or moves on.
    g.add_conditional_edges("fact_check", nodes.after_fact_check, ["researcher", "select"])
    # Conditional routing: if nothing new survived selection, end without sending an empty message.
    g.add_conditional_edges("select", nodes.after_select, {"writer": "writer", "end": END})
    g.add_edge("writer", "critic")
    # Quality loop: the critic sends the draft back to the writer until it passes (max 2 revisions).
    g.add_conditional_edges("critic", nodes.after_critic, {"writer": "writer", "send": "send"})
    g.add_edge("send", END)

    # --- Checkpointer: save the state after every node to data/checkpoints.sqlite. -------------------
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Explicitly allow our own state types to be restored from checkpoints (nothing else).
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("digest.state", name) for name in _STATE_TYPES]
    )
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=serde)
    # compile() turns the description above into a runnable graph (graph.invoke / graph.stream).
    return g.compile(checkpointer=checkpointer)
