"""Free page reading: download pages in parallel and extract their main text (no LLM involved)."""

from __future__ import annotations

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
