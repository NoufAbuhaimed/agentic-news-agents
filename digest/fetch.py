"""Free page reading: download pages in parallel and extract their main text (no LLM involved)."""

from __future__ import annotations

import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx
import trafilatura

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tech-digest/1.0; personal news digest)",
    "Accept": "text/html,application/xhtml+xml",
}


_HREF_RE = re.compile(r'href="([^"#]+)"', re.IGNORECASE)
_ISO_RE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MDY_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2}),? (20\d\d)\b", re.I)
_DMY_RE = re.compile(r"\b(\d{1,2}) (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,? (20\d\d)\b", re.I)


def page_dates(html: str, text: str) -> list[str]:
    """Every date on the page as YYYY-MM-DD: ISO dates in the HTML (incl. <time datetime=...>)
    plus written dates like 'Oct 5, 2026' or '5 October 2026' in the text."""
    found = {f"{y}-{m}-{d}" for y, m, d in _ISO_RE.findall(html)}
    for mon, day, year in _MDY_RE.findall(text):
        found.add(f"{year}-{_MONTHS[mon[:3].lower()]:02d}-{int(day):02d}")
    for day, mon, year in _DMY_RE.findall(text):
        found.add(f"{year}-{_MONTHS[mon[:3].lower()]:02d}-{int(day):02d}")
    return sorted(found)


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)  # absolute http(s) links found on the page
    dates: list[str] = field(default_factory=list)  # YYYY-MM-DD dates found on the page


def _fetch_one(url: str, max_chars: int) -> Page | None:
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            log.info("skip %s (%s)", url, r.status_code)
            return None
        text = trafilatura.extract(r.text, include_links=False, include_tables=True, favor_recall=True)
        if not text or len(text) < 200:
            log.info("skip %s (no readable text)", url)
            return None
        meta = trafilatura.extract_metadata(r.text)
        links = {urljoin(str(r.url), h) for h in _HREF_RE.findall(r.text)}
        return Page(
            url=str(r.url),
            title=(meta.title if meta and meta.title else ""),
            text=text[:max_chars],
            links=[u for u in links if u.startswith("http")],
            dates=page_dates(r.text, text),
        )
    except httpx.HTTPError as e:
        log.info("skip %s (%s)", url, e.__class__.__name__)
        return None


# --- Newsletter leads --------------------------------------------------------
# TLDR AI publishes an RSS feed of daily issues; each issue page links every story to its
# original source. We read it in code (no AI) and hand the stories to the planner as leads.
TLDR_AI_FEED = "https://tldr.tech/api/rss/ai"
_STORY_RE = re.compile(r'<a[^>]+href="(https?://[^"#]+)"[^>]*>(.*?)</a>', re.S)
_STORY_MARK = re.compile(r"\((\d+ minute read|GitHub Repo)\)", re.I)


def _clean_url(url: str) -> str:
    """Drop tracking parameters (utm_*), keep the rest."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(html.unescape(html.unescape(url)))
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def issue_stories(issue_html: str) -> list[dict]:
    """Stories from one newsletter issue: {headline, url}. Sponsors and non-story links are skipped."""
    stories, seen = [], set()
    for href, inner in _STORY_RE.findall(issue_html):
        text = html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()
        if "tldr.tech" in href or "(Sponsor)" in text or not _STORY_MARK.search(text):
            continue
        url = _clean_url(href)
        if url in seen:
            continue
        seen.add(url)
        stories.append({"headline": _STORY_MARK.sub("", text).strip(), "url": url})
    return stories


def newsletter_leads(since: str, max_issues: int = 3, feed_url: str = TLDR_AI_FEED) -> list[dict]:
    """Stories from newsletter issues published on/after `since` (YYYY-MM-DD).

    Never raises: if the feed or a page can't be read, the run simply continues without leads.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    try:
        feed = httpx.get(feed_url, headers=_HEADERS, timeout=15, follow_redirects=True)
        feed.raise_for_status()
        issues = []
        for item in ET.fromstring(feed.text).findall("./channel/item"):
            day = parsedate_to_datetime(item.findtext("pubDate")).date().isoformat()
            if day >= since:
                issues.append((day, item.findtext("link")))
        leads = []
        for day, link in sorted(issues, reverse=True)[:max_issues]:
            page = httpx.get(link, headers=_HEADERS, timeout=20, follow_redirects=True)
            if page.status_code == 200:
                leads += [{**s, "issue_date": day} for s in issue_stories(page.text)]
        log.info("newsletter: %d leads from %d issue(s) since %s", len(leads), len(issues[:max_issues]), since)
        return leads
    except Exception as e:  # network, feed format change, ...: leads are a bonus, never a blocker
        log.warning("newsletter leads unavailable (%s); continuing without them", e.__class__.__name__)
        return []


def _loads(url: str) -> bool:
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=10, follow_redirects=True)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def reachable(urls: list[str]) -> list[str]:
    """The subset of urls that load (HTTP 200), checked in parallel. Order is kept."""
    with ThreadPoolExecutor(max_workers=10) as pool:
        ok = list(pool.map(_loads, urls))
    return [u for u, good in zip(urls, ok) if good]


def fetch_pages(urls: list[str], max_chars: int) -> list[Page]:
    with ThreadPoolExecutor(max_workers=6) as pool:
        pages = pool.map(lambda u: _fetch_one(u, max_chars), urls)
    return [p for p in pages if p]
