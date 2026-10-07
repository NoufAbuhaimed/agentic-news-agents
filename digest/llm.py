"""Model access: Claude Haiku for web search, Claude Sonnet for writing, free OpenRouter models for the rest."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .config import settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Server-side refusal fallback for the 5.5 models: a declined request is re-routed inside the same call.
_FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "model_kwargs": {"fallbacks": "default"}}

MAX_CONTINUATIONS = 4


# --- Claude ------------------------------------------------------------------
# Claude Sonnet: writes the final Signal message (the part the group actually reads).
def writer_chat() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.writer_model,
        max_tokens=8000,
        reasoning_effort="medium",
        default_request_timeout=300,
        max_retries=3,
        **_FALLBACK,
    )


# Claude Haiku bound to Anthropic's server-side web_search tool: Anthropic runs the search.
def search_chat(max_uses: int) -> ChatAnthropic:
    """Haiku with Anthropic's basic web search tool: it only searches, it doesn't read pages."""
    return ChatAnthropic(
        model=settings.search_model,
        max_tokens=1000,
        default_request_timeout=300,
        max_retries=3,
    ).bind_tools([{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}])


# Last-resort brain for the research agents when the free models are busy or down.
def agent_fallback_chat() -> ChatAnthropic:
    """Claude Haiku as the research agent's last-resort model when free models are unavailable."""
    return ChatAnthropic(model=settings.search_model, max_tokens=4000, default_request_timeout=300, max_retries=2)


def run_search(messages: list[BaseMessage], max_uses: int) -> AIMessage:
    """Invoke the search model, continuing while the API pauses a long server-tool turn."""
    llm = search_chat(max_uses)
    messages = list(messages)
    response = llm.invoke(messages)
    for _ in range(MAX_CONTINUATIONS):
        if response.response_metadata.get("stop_reason") != "pause_turn":
            break
        messages.append(response)
        response = llm.invoke(messages)
    check_stop(response)
    return response


def search_results(response: AIMessage) -> list[dict]:
    """Pull {url, title, page_age} out of the web_search_tool_result blocks."""
    out: list[dict] = []
    for block in response.content if isinstance(response.content, list) else []:
        if isinstance(block, dict) and block.get("type") == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, list):  # a dict here is an error object (e.g. max_uses_exceeded)
                for r in content:
                    if r.get("url"):
                        out.append({"url": r["url"], "title": r.get("title", ""), "page_age": r.get("page_age")})
    return out


# Turn unusual stop reasons (refusal, output cut off) into a clear error/warning.
def check_stop(response: AIMessage) -> None:
    stop = response.response_metadata.get("stop_reason")
    if stop == "refusal":
        raise RuntimeError(f"model refused: {response.response_metadata.get('stop_details')}")
    if stop == "max_tokens":
        log.warning("response hit max_tokens; output may be truncated")


# --- Free models via OpenRouter ----------------------------------------------
# --- Free models via OpenRouter --------------------------------------------------------------
# OpenRouter exposes an OpenAI-compatible API, so we use LangChain's ChatOpenAI with its URL.
def free_chat(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        timeout=180,
        max_retries=1,  # fail fast so the fallback model takes over
        default_headers={"X-Title": "tech-digest"},
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# Find the JSON object in a model's reply and validate it against the Pydantic schema.
def _parse(text: str, schema: type[T]) -> T:
    match = _JSON_RE.search(text.replace("```json", "").replace("```", ""))
    if not match:
        raise ValueError("no JSON object in response")
    return schema.model_validate(json.loads(match.group(0)))


def free_json(messages: list[BaseMessage], schema: type[T]) -> T:
    """Ask a free model for JSON matching `schema`; on bad output or errors, try the next model.

    Free models don't reliably support structured-output APIs, so the schema goes in the prompt
    and the reply is validated with pydantic.
    """
    instructions = HumanMessage(
        "Reply with only one JSON object matching this JSON Schema, no prose and no code fences:\n"
        + json.dumps(schema.model_json_schema())
    )
    errors = []
    for model in settings.free_models:
        for attempt in range(2):
            try:
                reply = free_chat(model).invoke(list(messages) + [instructions])
                return _parse(reply.text, schema)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                errors.append(f"{model}: bad output ({e.__class__.__name__})")
                log.warning("free model %s returned unusable output (attempt %d)", model, attempt + 1)
            except Exception as e:  # rate limits, provider outages, model removed
                errors.append(f"{model}: {e.__class__.__name__}: {str(e)[:120]}")
                log.warning("free model %s failed: %s", model, str(e)[:160])
                break  # move on to the next model
    # Last resort: Claude Haiku (paid, ~1 cent per call), so a day where the free models are
    # rate-limited or down doesn't stop the run.
    try:
        reply = agent_fallback_chat().invoke(list(messages) + [instructions])
        log.warning("free models unavailable (%s); used Claude Haiku instead", "; ".join(errors)[:200])
        return _parse(reply.text, schema)
    except Exception as e:
        errors.append(f"claude-haiku fallback: {e.__class__.__name__}: {str(e)[:120]}")
    raise RuntimeError("all free models failed: " + " | ".join(errors))
