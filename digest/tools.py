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


@dataclass
class ResearchContext:
    thread: ResearchThread
    today: str
    since: str
    already_sent: set[str]
    pages: dict[str, Page] = field(default_factory=dict)  # normalized URL -> fetched page
    findings: list[Candidate] = field(default_factory=list)
    # URLs the agent has actually seen: starting points, search results, links on fetched pages.
    # fetch_page only opens these, so the agent searches instead of guessing URLs from memory.
    known_urls: set[str] = field(default_factory=set)

    def learn(self, urls) -> None:
        self.known_urls.update(normalize_url(u) for u in urls)


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _before_window(published: str, since: str) -> bool:
    try:
        return date.fromisoformat(published[:10]) < date.fromisoformat(since)
    except ValueError:
        return False  # "unknown": the editor decides later


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


def build_tools(ctx: ResearchContext) -> list:
    @tool
    def web_search(query: str) -> str:
        """Search the web. Returns titles, URLs and page ages. Use specific queries, e.g.
        'LangGraph release October 2026' or 'MCP specification changelog'."""
        if costs.over_budget():
            return "The search budget for this run is used up. Don't search again; save what you have and finish."
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
        results = search_results(response)[:8]
        ctx.learn(r["url"] for r in results)
        if not results:
            return "No results. Try a different query."
        return "\n".join(f"- {r['title']} | {r['url']} | {r.get('page_age') or 'date unknown'}" for r in results)

    @tool
    def fetch_page(url: str) -> str:
        """Download a web page and return its main text. Use it on official sources (release
        pages, changelogs, blog posts) before saving a finding from them."""
        key = normalize_url(url)
        if key in ctx.pages:
            return "You already fetched this page. Use what you read, or fetch a different page."
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
        return f"{page.title}\n{page.url}\n\n{page.text[: settings.agent_page_chars]}"

    @tool
    def check_already_sent(url: str) -> str:
        """Check the digest's memory: was this URL already posted in a previous digest?"""
        if normalize_url(url) in ctx.already_sent:
            return "Already posted in a previous digest. Skip it."
        return "Not posted before."

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
        page = ctx.pages.get(normalize_url(source_page_url))
        if page is None:
            return "Rejected: you haven't fetched source_page_url. Fetch it first, then save."
        if len(evidence_quote.strip()) < 20 or _squash(evidence_quote) not in _squash(page.text):
            return "Rejected: evidence_quote must be copied exactly from the fetched page (at least one full sentence)."
        published = published.strip()[:10]
        if published not in page.dates:
            return (
                f"Rejected: the date '{published}' doesn't appear on that page. Use the item's publication "
                "date exactly as shown on the page, as YYYY-MM-DD. Skip items whose date you can't see."
            )
        if _before_window(published, ctx.since):
            return f"Rejected: published {published} is before the window ({ctx.since})."
        if _host(url) != _host(page.url):
            url = page.url  # only trust URLs on the page we actually read
        key = normalize_url(url)
        if key in ctx.already_sent:
            return "Rejected: already posted in a previous digest."
        if any(normalize_url(f.url) == key and f.title == title for f in ctx.findings):
            return "Already saved."
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
