"""The research agent: a create_agent loop with four tools and guard-rail middleware.

The model decides which tool to call next and stops by answering without a tool call.
Guard rails keep a confused model from looping or overspending.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage

from . import prompts
from .config import settings
from .llm import agent_fallback_chat, free_chat
from .tools import ResearchContext, build_tools

log = logging.getLogger(__name__)


# --- Guard rail 1: loop detection (our own middleware) ------------------------------------------
# Middleware = code that runs around every model turn or tool call of the agent.
class LoopDetector(AgentMiddleware):
    """Catches an agent repeating itself.

    The first repeat of an identical tool call (same tool, same arguments) is answered with a
    warning instead of running it again. After `max_repeats` repeats, the agent is stopped.
    """

    def __init__(self, label: str, max_repeats: int = 3):
        super().__init__()
        self.label = label
        self.max_repeats = max_repeats
        self.seen: Counter[str] = Counter()
        self.repeats = 0

    # Runs BEFORE each tool call. `handler(request)` would actually run the tool.
    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        # Identity of a call = tool name + its exact arguments.
        key = f"{call['name']}:{json.dumps(call['args'], sort_keys=True)}"
        self.seen[key] += 1
        if self.seen[key] > 1:
            self.repeats += 1
            log.warning("loop detector[%s]: repeated %s call (%d repeats so far)", self.label, call["name"], self.repeats)
            return ToolMessage(
                content=(
                    f"Loop detected: you already called {call['name']} with exactly these arguments. "
                    "Don't repeat it. Try something different, or finish if you have enough."
                ),
                tool_call_id=call["id"],
                status="error",
            )
        # Not a repeat: run the tool normally.
        return handler(request)

    # Runs BEFORE each model turn. Returning {"jump_to": "end"} ends the agent loop immediately.
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if self.repeats >= self.max_repeats:
            log.warning("loop detector[%s]: %d repeats, stopping the agent", self.label, self.repeats)
            return {"jump_to": "end", "messages": [AIMessage("Stopped by the loop detector.")]}
        return None


# --- Guard rail 2: time limit (our own middleware) -----------------------------------------------
# Caps how long one agent may work (4 min, 2 on a retry), so one slow agent can't hold up the run.
class TimeLimit(AgentMiddleware):
    """Stops the agent once its time is up; findings saved so far are kept."""

    def __init__(self, label: str, seconds: float):
        super().__init__()
        self.label = label
        self.seconds = seconds
        self.started: float | None = None

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        now = time.monotonic()
        if self.started is None:
            self.started = now
        elif now - self.started > self.seconds:
            log.info("time limit[%s]: %.0fs used, stopping the agent", self.label, now - self.started)
            return {"jump_to": "end", "messages": [AIMessage("Stopped: time limit reached.")]}
        return None


# --- Building the agent ------------------------------------------------------------------------
# create_agent (LangChain) builds the agent loop as a small LangGraph graph:
#     model node ─(tool call)→ tools node ─(result)→ model node … ─(no tool call)→ end
# The model chooses the tools; the middleware list below is applied around every step.
def build_research_agent(
    ctx: ResearchContext, search_limit: int, max_steps: int | None = None, time_limit_s: float | None = None
):
    models = list(settings.free_models)
    return create_agent(
        # The agent's "brain": a free OpenRouter model (NVIDIA Nemotron by default).
        free_chat(models[0]),
        # The 4 tools (tools.py), bound to this agent's private notebook `ctx`.
        build_tools(ctx),
        # How to work: verify on official pages, stay in the date window, stop when done.
        system_prompt=prompts.AGENT,
        middleware=[
            # Free model busy or down → next free model → Claude Haiku.
            ModelFallbackMiddleware(*[free_chat(m) for m in models[1:2]], agent_fallback_chat()),
            # Hard limits: total model turns, and per-tool call counts.
            ModelCallLimitMiddleware(run_limit=max_steps or settings.agent_max_steps, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=search_limit, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="fetch_page", run_limit=settings.pages_per_thread, exit_behavior="continue"),
            # Our own guard rails (defined above).
            LoopDetector(ctx.thread.name),
            TimeLimit(ctx.thread.name, time_limit_s or settings.agent_time_limit_s),
        ],
        name="research_agent",
    )
