"""The research agent: a create_agent loop with four tools and guard-rail middleware.

The model decides which tool to call next and stops by answering without a tool call.
Guard rails keep a confused model from looping or overspending.
"""

from __future__ import annotations

import json
import logging
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

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
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
        return handler(request)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if self.repeats >= self.max_repeats:
            log.warning("loop detector[%s]: %d repeats, stopping the agent", self.label, self.repeats)
            return {"jump_to": "end", "messages": [AIMessage("Stopped by the loop detector.")]}
        return None


def build_research_agent(ctx: ResearchContext, search_limit: int, max_steps: int | None = None):
    models = list(settings.free_models)
    return create_agent(
        free_chat(models[0]),
        build_tools(ctx),
        system_prompt=prompts.AGENT,
        middleware=[
            # Free model busy or down → next free model → Claude Haiku.
            ModelFallbackMiddleware(*[free_chat(m) for m in models[1:2]], agent_fallback_chat()),
            # Hard limits: total model turns, and per-tool call counts.
            ModelCallLimitMiddleware(run_limit=max_steps or settings.agent_max_steps, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=search_limit, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="fetch_page", run_limit=settings.pages_per_thread, exit_behavior="continue"),
            LoopDetector(ctx.thread.name),
        ],
        name="research_agent",
    )
