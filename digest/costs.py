"""Per-run cost tracking and the hard budget cap.

A LangChain callback sees every model call in the run, including the ones made inside tools and
inside each research agent, and adds up the cost from the token usage the API reports.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Free OpenRouter models (":free") cost nothing.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
WEB_SEARCH_USD = 0.01  # per search request


# Look up a model's price; free OpenRouter models (":free") cost nothing.
def _price(model: str) -> tuple[float, float] | None:
    if model.endswith(":free"):
        return (0.0, 0.0)
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return None


# A LangChain callback: after EVERY model call (including inside tools and agents) LangChain calls
# on_llm_end(), where we read the token counts and number of searches and add up the cost.
class CostTracker(BaseCallbackHandler):
    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self.total_usd = 0.0
        self.by_model: dict[str, float] = defaultdict(float)
        self.searches = 0
        self._warned = False

    def on_llm_end(self, response, **kwargs) -> None:
        for gen in (g for gens in response.generations for g in gens):
            msg = getattr(gen, "message", None)
            if msg is None:
                continue
            meta = msg.response_metadata or {}
            model = meta.get("model") or meta.get("model_name") or "unknown"
            usage = msg.usage_metadata or {}
            searches = ((meta.get("usage") or {}).get("server_tool_use") or {}).get("web_search_requests", 0) or 0
            price = _price(model)
            if price is None:
                log.warning("no price for model %s; counting it as free", model)
                price = (0.0, 0.0)
            cost = (
                usage.get("input_tokens", 0) / 1e6 * price[0]
                + usage.get("output_tokens", 0) / 1e6 * price[1]
                + searches * WEB_SEARCH_USD
            )
            with self._lock:
                self.total_usd += cost
                self.by_model[model] += cost
                self.searches += searches
                if self.over_budget and not self._warned:
                    self._warned = True
                    log.warning("run budget of $%.2f reached ($%.2f); no more searches", self.budget_usd, self.total_usd)

    @property
    def over_budget(self) -> bool:
        return self.total_usd >= self.budget_usd

    def summary(self) -> str:
        parts = ", ".join(f"{m} ${c:.3f}" for m, c in sorted(self.by_model.items(), key=lambda x: -x[1]) if c)
        return f"${self.total_usd:.3f} (searches={self.searches}; {parts or 'all free'})"


# One tracker per run, set by the CLI. Tools and routing read it to enforce the budget.
_current: CostTracker | None = None
_search_down: str | None = None  # set when search fails in a way retrying won't fix (credits, auth)


def start(budget_usd: float) -> CostTracker:
    global _current, _search_down
    _current = CostTracker(budget_usd)
    _search_down = None
    return _current


def disable_search(reason: str) -> None:
    global _search_down
    if _search_down is None:
        log.error("web search disabled for the rest of this run: %s", reason)
    _search_down = reason


def search_down() -> str | None:
    return _search_down


def over_budget() -> bool:
    return _current is not None and _current.over_budget
