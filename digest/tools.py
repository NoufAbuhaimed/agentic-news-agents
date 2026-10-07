"""The research agent's tools. Each agent gets its own set, bound to a ResearchContext that holds
the pages it fetched and the findings it saved, so `save_finding` can verify against real pages."""

from __future__ import annotations

import logging
import re

import anthropic
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from . import costs, prompts
from .config import settings
from .fetch import Page, fetch_pages
from .llm import run_search, search_results
from .state import Candidate, ResearchThread
from .store import normalize_url

log = logging.getLogger(__name__)


# --- The agent's private notebook --------------------------------------------------------------
# Each research agent gets its own ResearchContext. The tools below read and write it; that is how
# plain code can check the agent's claims (e.g. "was this page really fetched?").
@dataclass
class ResearchContext:
    thread: ResearchThread
    today: str
    since: str
    already_sent: set[str]
    # Full text, links and dates of every page this agent opened (used to verify save_finding).
    pages: dict[str, Page] = field(default_factory=dict)  # normalized URL -> fetched page
    # Verified news items saved by this agent; returned to the graph as `candidates`.
    findings: list[Candidate] = field(default_factory=list)
    # URLs the agent has actually seen: starting points, search results, links on fetched pages.
    # fetch_page only opens these, so the agent searches instead of guessing URLs from memory.
    known_urls: set[str] = field(default_factory=set)

    # Add URLs to the "allowed to open" list (normalized so trivial differences don't matter).
    def learn(self, urls) -> None:
        self.known_urls.update(normalize_url(u) for u in urls)


# Small helpers used by the checks below.
def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


# Lower-case and collapse whitespace, so a quote matches even if line breaks differ.
def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _before_window(published: str, since: str) -> bool:
    try:
        return date.fromisoformat(published[:10]) < date.fromisoformat(since)
    except ValueError:
        return False  # "unknown": the editor decides later


# Message returned to the agent when search can't work this run (e.g. out of credits).
_SEARCH_DOWN = (
    "Web search is unavailable for this run. Don't call web_search again. Use fetch_page on your "
    "starting URLs and the links on pages you read, then save what you can verify."
)


def source_context(page_text: str, quote: str, radius: int = 2500) -> str:
    """The section of the page around the quote (whitespace-normalized), for fact-checking."""
    flat, q = re.sub(r"\s+", " ", page_text), _squash(quote)
    at = flat.lower().find(q)
    if at < 0:
        return flat[: 2 * radius]
    return flat[max(0, at - radius): at + len(q) + radius]


# --- The tools ---------------------------------------------------------------------------------
# Tool integration: four @tool functions. The model sees each tool's NAME and DOCSTRING and decides
# when to call it. They are created inside build_tools() so each one is bound to this agent's `ctx`.
# A tool never raises to the agent: problems come back as a short text the model can react to.
def build_tools(ctx: ResearchContext) -> list:
    # Tool 1: search the live internet (Anthropic web search via Claude Haiku). The only tool that costs money.
    @tool
    def web_search(query: str) -> str:
        """Search the web. Returns titles, URLs and page ages. Use specific queries, e.g.
        'LangGraph release October 2026' or 'MCP specification changelog'."""
        # Budget guard rail: once the run's $ cap is reached, searches stop.
        if costs.over_budget():
            return "The search budget for this run is used up. Don't search again; save what you have and finish."
        # Graceful degradation: if search failed for good earlier (e.g. no credits), don't keep trying.
        if costs.search_down():
            return _SEARCH_DOWN
        try:
            response = run_search(
                [SystemMessage(prompts.SEARCH), HumanMessage(f"Search for: {query}")], max_uses=1
            )
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 401, 403):  # e.g. no credits, bad key: won't fix itself this run
                costs.disable_search(f"{e.status_code}: {getattr(e, 'message', e)}"[:200])
                return _SEARCH_DOWN
            log.warning("web_search failed (%s); agent continues", e.status_code)
            return "Search failed this time. Try again later or use fetch_page on the URLs you already have."
        except anthropic.APIConnectionError:
            log.warning("web_search connection error; agent continues")
            return "Search failed this time. Try again later or use fetch_page on the URLs you already have."
        # Search results become URLs the agent is allowed to open with fetch_page.
        results = search_results(response)[:8]
        ctx.learn(r["url"] for r in results)
        if not results:
            return "No results. Try a different query."
        return "\n".join(f"- {r['title']} | {r['url']} | {r.get('page_age') or 'date unknown'}" for r in results)

    # Tool 2: open one web page and return its text (plain code, free).
    @tool
    def fetch_page(url: str) -> str:
        """Download a web page and return its main text. Use it on official sources (release
        pages, changelogs, blog posts) before saving a finding from them."""
        key = normalize_url(url)
        if key in ctx.pages:
            return "You already fetched this page. Use what you read, or fetch a different page."
        # Anti-hallucination rule: only URLs the agent has actually SEEN may be opened, never guessed ones.
        if key not in ctx.known_urls:
            return (
                "Not allowed: you can only open URLs from your starting list, search results, or links on "
                "pages you've read. Don't guess URLs; use web_search to find the page first."
            )
        pages = fetch_pages([url], settings.page_max_chars)
        if not pages:
            return "Could not read this page (blocked, not HTML, or empty). Try another source."
        page = pages[0]
        ctx.pages[key] = page
        ctx.pages[normalize_url(page.url)] = page
        ctx.learn(page.links)
        # Links on this page become openable too (e.g. a releases list → one specific release).
        return f"{page.title}\n{page.url}\n\n{page.text[: settings.agent_page_chars]}"

    # Tool 3: long-term memory lookup (links posted in earlier digests).
    @tool
    def check_already_sent(url: str) -> str:
        """Check the digest's memory: was this URL already posted in a previous digest?"""
        if normalize_url(url) in ctx.already_sent:
            return "Already posted in a previous digest. Skip it."
        return "Not posted before."

    # Tool 4: save a news item, but ONLY if plain code can verify it. This is the heart of the
    # "no hallucinations" design: every check below compares the agent's claim with the real page.
    @tool
    def save_finding(
        title: str,
        url: str,
        source_page_url: str,
        published: str,
        summary: str,
        why_it_matters: str,
        evidence_quote: str,
    ) -> str:
        """Save one verified news item.

        Args:
            title: Short headline.
            url: Most specific URL for the item (a release tag, the post itself).
            source_page_url: The page you fetched that this item comes from.
            published: Publication date as YYYY-MM-DD; it must be shown on the source page.
            summary: What happened, 1-2 factual sentences.
            why_it_matters: Why an agent builder should care, 1 sentence.
            evidence_quote: A sentence copied exactly from the fetched page that supports the summary.
        """
        # Check 1: the source page must really have been fetched by this agent.
        page = ctx.pages.get(normalize_url(source_page_url))
        if page is None:
            return "Rejected: you haven't fetched source_page_url. Fetch it first, then save."
        # Check 2: the evidence quote must appear word for word on that page (no invented quotes).
        if len(evidence_quote.strip()) < 20 or _squash(evidence_quote) not in _squash(page.text):
            return "Rejected: evidence_quote must be copied exactly from the fetched page (at least one full sentence)."
        published = published.strip()[:10]
        # Check 3: the publication date must be shown on the page (no invented dates).
        if published not in page.dates:
            return (
                f"Rejected: the date '{published}' doesn't appear on that page. Use the item's publication "
                "date exactly as shown on the page, as YYYY-MM-DD. Skip items whose date you can't see."
            )
        # Check 4: only news inside the date window.
        if _before_window(published, ctx.since):
            return f"Rejected: published {published} is before the window ({ctx.since})."
        # Check 5: a link to a different website than the page we read is replaced by that page.
        if _host(url) != _host(page.url):
            url = page.url  # only trust URLs on the page we actually read
        key = normalize_url(url)
        # Check 6: never repeat news from earlier digests (memory), or save the same item twice.
        if key in ctx.already_sent:
            return "Rejected: already posted in a previous digest."
        if any(normalize_url(f.url) == key and f.title == title for f in ctx.findings):
            return "Already saved."
        # All checks passed: store the item, plus ~5,000 characters of the page around the quote
        # (`context`) so the fact_check node can later verify the whole summary.
        ctx.findings.append(
            Candidate(
                title=title,
                url=url,
                published=published,
                summary=summary,
                why_it_matters=why_it_matters,
                evidence=evidence_quote,
                context=source_context(page.text, evidence_quote),
                verified=True,
                thread=ctx.thread.name,
            )
        )
        return f"Saved ({len(ctx.findings)} so far)."

    return [web_search, fetch_page, check_already_sent, save_finding]
