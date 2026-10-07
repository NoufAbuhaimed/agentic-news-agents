"""Local logging (stdout + rotating file) and a LangChain callback that logs each graph step.

LangSmith tracing needs no code here: set LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT
and LangGraph sends every node, model call and web search to your project automatically.
"""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler

import httpx
from langchain_core.callbacks import BaseCallbackHandler

from .config import settings

log = logging.getLogger("digest")


# Log to the terminal and to data/logs/digest.log (rotated, so it never grows without limit).
def setup_logging() -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(getattr(h, "_digest", False) for h in root.handlers):
        return  # already set up (e.g. re-running a notebook cell)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root.setLevel(logging.INFO)
    for handler in (
        logging.StreamHandler(),
        RotatingFileHandler(settings.log_dir / "digest.log", maxBytes=5_000_000, backupCount=5),
    ):
        handler.setFormatter(fmt)
        handler._digest = True
        root.addHandler(handler)
    for noisy in ("httpx", "httpx2", "trafilatura"):
        logging.getLogger(noisy).setLevel(logging.WARNING if noisy != "trafilatura" else logging.CRITICAL)


_MAIN_NODES = {"planner", "researcher", "fact_check", "select", "writer", "critic", "send"}


# LangChain callbacks are notified about every chain/node, model call and tool call in the run.
# We use them to print the readable step log: ▶ node, 🔧 tool(args), ↳ result, llm tokens.
class StepLogger(BaseCallbackHandler):
    """One log line per main graph node, per model call (tokens, searches) and per agent tool call."""

    def __init__(self):
        self._starts: dict = {}
        self._tools: dict = {}

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name", "tool")
        self._tools[run_id] = name
        log.info("  🔧 %s(%s)", name, str(inputs or input_str)[:150])

    def on_tool_end(self, output, *, run_id, **kw):
        name = self._tools.pop(run_id, "tool")
        text = getattr(output, "content", output)
        log.info("     ↳ %s: %s", name, str(text).replace("\n", " ")[:120])

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kw):
        node = (metadata or {}).get("langgraph_node")
        if node in _MAIN_NODES and kw.get("name") == node:
            self._starts[run_id] = (node, time.monotonic())
            log.info("▶ %s", node)

    def on_chain_end(self, outputs, *, run_id, **kw):
        if run_id in self._starts:
            node, t0 = self._starts.pop(run_id)
            log.info("✔ %s (%.1fs)", node, time.monotonic() - t0)

    def on_chain_error(self, error, *, run_id, **kw):
        if run_id in self._starts:
            node, _ = self._starts.pop(run_id)
            log.error("✘ %s: %s", node, error)

    def on_llm_end(self, response, **kw):
        for gen in (g for gens in response.generations for g in gens):
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "usage_metadata", None) or {}
            usage = (getattr(msg, "response_metadata", None) or {}).get("usage", {}) or {}
            server = usage.get("server_tool_use") or {}
            log.info(
                "  llm %s: in=%s out=%s searches=%s fetches=%s",
                ((msg.response_metadata or {}).get("model") or (msg.response_metadata or {}).get("model_name", "?")) if msg else "?",
                meta.get("input_tokens"),
                meta.get("output_tokens"),
                server.get("web_search_requests", 0),
                server.get("web_fetch_requests", 0),
            )


class RootRunCapture(BaseCallbackHandler):
    """Remembers the top-level run id, so we can link to this run's trace in LangSmith."""

    def __init__(self):
        self.run_id = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        if parent_run_id is None and self.run_id is None:
            self.run_id = run_id


# Optional external health check (e.g. healthchecks.io emails you if a run fails or is missed).
def ping(suffix: str = "") -> None:
    """healthchecks.io-style ping: '' = success, '/start', '/fail'. Never raises."""
    if not settings.healthcheck_url:
        return
    try:
        httpx.get(settings.healthcheck_url.rstrip("/") + suffix, timeout=10)
    except httpx.HTTPError as e:
        log.warning("healthcheck ping failed: %s", e)
