"""Graph nodes: planner → research agents (parallel) → fact_check (⟲ retry) → select → writer ⇄ critic → send.

Who does what (to keep cost low):
  planner, research-agent reasoning, fact_check, select, critic → free OpenRouter models
                                                    (agents fall back to Claude Haiku)
  web search (a tool the agents call)              → Claude Haiku + Anthropic web search
  fetching pages (a tool the agents call)          → plain Python (free)
  writing the Signal message                       → Claude Sonnet
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import anthropic

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from . import costs, prompts, signal_client
from .agent import build_research_agent
from .config import settings
from .fetch import reachable
from .llm import check_stop, free_chat, free_json, writer_chat
from .state import (
    Candidate,
    Critique,
    DigestState,
    FactCheck,
    Plan,
    ResearcherInput,
    Selection,
)
from .tools import ResearchContext
from .store import Store, normalize_url

log = logging.getLogger(__name__)


@lru_cache
def store() -> Store:
    return Store(settings.db_path)


def _items_block(items: list[Candidate]) -> str:
    fields = ("title", "url", "published", "summary", "why_it_matters")
    return json.dumps([c.model_dump(include=set(fields)) for c in items], indent=1, ensure_ascii=False)


# --- planner ---------------------------------------------------------------
def planner(state: DigestState) -> dict:
    remembered = store().top_sources()
    memory = f"\nSources that produced published items before: {', '.join(remembered)}." if remembered else ""
    plan = free_json(
        [
            SystemMessage(prompts.PLANNER),
            HumanMessage(f"Today is {state['today']}. Cover news published on or after {state['since']}.{memory}"),
        ],
        Plan,
    )
    threads = plan.threads[:4]
    # The planner writes URLs from memory; only pass on the ones that actually load.
    proposed = sum(len(t.starting_points) for t in threads)
    for t in threads:
        t.starting_points = reachable(t.starting_points)
    kept = sum(len(t.starting_points) for t in threads)
    log.info("planned %d threads: %s (%d/%d starting URLs load)",
             len(threads), ", ".join(t.name for t in threads), kept, proposed)
    return {"threads": threads}


def _send_research(state: DigestState, thread, attempt: int = 1, feedback: str = "") -> Send:
    return Send(
        "researcher",
        {
            "thread": thread,
            "today": state["today"],
            "since": state["since"],
            "recent_urls": state.get("recent_urls", []),
            "attempt": attempt,
            "feedback": feedback,
        },
    )


def fan_out(state: DigestState) -> list[Send]:
    return [_send_research(state, t) for t in state["threads"]]


# --- research agent (one per topic, in parallel) -------------------------------
def researcher(inp: ResearcherInput) -> dict:
    thread = inp["thread"]
    attempt = inp.get("attempt", 1)
    ctx = ResearchContext(
        thread=thread, today=inp["today"], since=inp["since"], already_sent=set(inp.get("recent_urls", []))
    )
    ctx.learn(list(thread.starting_points) + prompts.SEED_URLS)
    task = (
        f"Today is {inp['today']}. Date window: {inp['since']} to {inp['today']}.\n"
        f"Your topic: {thread.name}\nFocus: {thread.focus}\n\n"
        "Starting URLs for this topic:\n" + "\n".join(f"- {u}" for u in thread.starting_points)
        + "\n\nGeneral sources:\n" + "\n".join(f"- {u}" for u in prompts.SEED_URLS)
    )
    if inp.get("feedback"):
        task += f"\n\nThis is retry #{attempt - 1}. Feedback from the fact-checker:\n{inp['feedback']}"

    # Retries get a smaller search allowance so a weak topic can't double the run's cost.
    search_limit = settings.search_max_uses if attempt == 1 else settings.retry_search_max_uses
    max_steps = settings.agent_max_steps if attempt == 1 else settings.retry_max_steps
    time_limit = settings.agent_time_limit_s if attempt == 1 else settings.retry_time_limit_s
    task += f"\n\nBudget: at most {search_limit} web searches and {settings.pages_per_thread} page fetches."
    agent = build_research_agent(ctx, search_limit, max_steps, time_limit)
    try:
        # Each agent turn is ~8 graph steps (model, tools, middleware hooks); our own limits stop it first.
        result = agent.invoke({"messages": [HumanMessage(task)]}, {"recursion_limit": 10 * settings.agent_max_steps + 10})
        final = result["messages"][-1].text[:200] if result.get("messages") else ""
    except Exception as e:
        if not ctx.findings:
            raise  # nothing to salvage: let the node's RetryPolicy / checkpoint resume handle it
        final = f"agent error after saving findings: {e.__class__.__name__}"
        log.warning("research agent[%s] failed after %d findings; keeping them: %s", thread.name, len(ctx.findings), e)

    log.info(
        "research agent[%s] attempt %d: fetched %d pages, saved %d findings. Final: %s",
        thread.name, attempt, len({p.url for p in ctx.pages.values()}), len(ctx.findings), final,
    )
    return {"candidates": ctx.findings, "attempts": {thread.name: attempt}}


# --- fact_check (⟲ can send a weak topic back to its research agent) --------------
def _key(c: Candidate) -> str:
    return f"{normalize_url(c.url)}|{c.title}"


def fact_check(state: DigestState) -> dict:
    done = set(state.get("fact_checked", []))
    new = [c for c in state.get("candidates", []) if _key(c) not in done]
    if not new:
        return {}

    numbered = "\n\n".join(
        f"[{i}] Title: {c.title}\nSummary: {c.summary}\nSource excerpt: {c.context or c.evidence}"
        for i, c in enumerate(new)
    )
    supported, notes = list(new), {}
    try:
        verdicts = {v.index: v for v in free_json([SystemMessage(prompts.FACT_CHECK), HumanMessage(numbered)], FactCheck).verdicts}
        supported = [c for i, c in enumerate(new) if verdicts.get(i) is None or verdicts[i].supported]
        for i, c in enumerate(new):
            v = verdicts.get(i)
            if v is not None and not v.supported:
                notes.setdefault(c.thread, []).append(f"'{c.title}' was rejected: {v.problem}")
    except RuntimeError as e:
        # Evidence quotes were already verified against the pages in code; keep them.
        log.warning("fact_check model unavailable, keeping code-verified items: %s", e)

    log.info("fact_check: %d new, %d supported, %d rejected", len(new), len(supported), len(new) - len(supported))
    return {"checked": supported, "fact_checked": [_key(c) for c in new], "fact_notes": notes}


def after_fact_check(state: DigestState):
    """Send topics with no verified findings back to research once, with feedback."""
    if costs.over_budget():
        log.warning("budget reached; no research retries")
        return "select"
    have_total = len(state.get("checked", []))
    if have_total >= settings.enough_items:
        log.info("fact_check: %d verified items is enough; skipping retries", have_total)
        return "select"
    retries = []
    for t in state["threads"]:
        have = sum(1 for c in state.get("checked", []) if c.thread == t.name)
        attempt = state.get("attempts", {}).get(t.name, 1)
        if have == 0 and attempt < settings.max_research_attempts:
            problems = state.get("fact_notes", {}).get(t.name, [])
            feedback = (
                "No verified findings yet. Try different, more specific searches and official release "
                "pages or changelogs, and make sure items fall inside the date window."
                + ("\nProblems found:\n- " + "\n- ".join(problems) if problems else "")
            )
            retries.append(_send_research(state, t, attempt + 1, feedback))
    if retries:
        log.info("fact_check: sending %d topic(s) back to research", len(retries))
        return retries
    return "select"


# --- select: dedupe in code, then let the editor model rank -------------------
def select(state: DigestState) -> dict:
    already_sent = set(state.get("recent_urls", []))
    seen: set[str] = set()
    fresh: list[Candidate] = []
    for c in state.get("checked", []):
        if not c.verified:
            continue  # never publish an item that doesn't trace back to a page we read
        key = normalize_url(c.url)
        if key in already_sent or key in seen:
            continue
        seen.add(key)
        fresh.append(c)
    log.info("%d fact-checked items, %d new", len(state.get("checked", [])), len(fresh))

    if not fresh:
        return {"selected": []}

    numbered = "\n".join(
        f"[{i}] {json.dumps(c.model_dump(include={'title', 'url', 'published', 'summary', 'why_it_matters'}), ensure_ascii=False)}"
        for i, c in enumerate(fresh)
    )
    choice = free_json(
        [
            SystemMessage(prompts.SELECT.format(max_items=settings.max_items)),
            HumanMessage(f"Date window: {state['since']} to {state['today']}.\n\n{numbered}"),
        ],
        Selection,
    )
    picked = [fresh[i] for i in dict.fromkeys(choice.chosen) if 0 <= i < len(fresh)][: settings.max_items]
    log.info("selected %d of %d: %s", len(picked), len(fresh), choice.reasoning)
    return {"selected": picked}


def after_select(state: DigestState) -> str:
    if not state.get("selected"):
        log.warning("nothing new to report; skipping send")
        return "end"
    return "writer"


# --- writer (Claude Sonnet) ⇄ critic (free model) -------------------------------
def writer(state: DigestState) -> dict:
    ask = f"Items:\n{_items_block(state['selected'])}"
    critique = state.get("critique")
    if critique and not critique.passed:
        ask += (
            f"\n\nYour previous draft:\n{state['digest']}\n\n"
            "Fix these problems:\n- " + "\n- ".join(critique.problems)
        )
    messages = [
        SystemMessage(prompts.WRITER.format(today=state["today"], max_chars=settings.max_chars)),
        HumanMessage(ask),
    ]
    try:
        response = writer_chat().invoke(messages)
        check_stop(response)
        text = response.text
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        # Claude unavailable (credits, outage): a free model writes the digest instead.
        log.warning("writer: Claude unavailable (%s); using a free model", e.__class__.__name__)
        text = ""
        for model in settings.free_models:
            try:
                text = free_chat(model).invoke(messages).text
                break
            except Exception as fe:  # try the next free model
                log.warning("writer: %s failed: %s", model, str(fe)[:120])
        if not text:
            raise
    return {"digest": text.strip(), "revisions": state.get("revisions", 0) + 1}


_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def critic(state: DigestState) -> dict:
    digest = state["digest"]
    problems: list[str] = []

    # Deterministic checks first: cheap and exact.
    if len(digest) > settings.max_chars:
        problems.append(f"Digest is {len(digest)} characters; cut it to under {settings.max_chars}.")
    allowed = {normalize_url(c.url) for c in state["selected"]}
    used = {normalize_url(u.rstrip(".,")) for u in _URL_RE.findall(digest)}
    if unknown := used - allowed:
        problems.append(f"These URLs are not from the items; use the item URLs exactly: {sorted(unknown)}")

    try:
        review = free_json(
            [
                SystemMessage(prompts.CRITIC),
                HumanMessage(f"Items:\n{_items_block(state['selected'])}\n\nDigest:\n{digest}"),
            ],
            Critique,
        )
        problems += review.problems if not review.passed else []
    except RuntimeError as e:
        log.warning("model review skipped (free models unavailable): %s", e)

    result = Critique(passed=not problems, problems=problems)
    log.info("critic: %s", "pass" if result.passed else f"{len(problems)} problem(s): {problems}")
    return {"critique": result}


def after_critic(state: DigestState) -> str:
    if state["critique"].passed:
        return "send"
    if state.get("revisions", 0) > settings.max_revisions:
        log.warning("critic still unhappy after %d revisions; sending last draft. Problems: %s",
                    settings.max_revisions, state["critique"].problems)
        return "send"
    return "writer"


# --- send ------------------------------------------------------------------
def send(state: DigestState) -> dict:
    if state.get("dry_run"):
        log.info("dry run — not sending")
        return {"sent": False}
    signal_client.send(state["digest"])
    store().record_sent([(c.url, c.title) for c in state["selected"]])
    log.info("sent digest with %d items", len(state["selected"]))
    return {"sent": True}
