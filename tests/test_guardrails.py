"""Guard rails around the research agent: the tools' verification rules and the loop detector."""

import pytest
from langchain_core.messages import ToolMessage

from digest.agent import LoopDetector
from digest.fetch import Page, page_dates
from digest.state import ResearchThread
from digest.store import normalize_url
from digest.tools import ResearchContext, build_tools

SOURCE = "https://github.com/acme/agentkit/releases"
QUOTE = "Adds parallel tool calls to the agent runtime."


@pytest.fixture
def ctx():
    c = ResearchContext(
        thread=ResearchThread(name="Frameworks", focus="f"),
        today="2026-10-07",
        since="2026-10-04",
        already_sent={normalize_url("https://old.example.com/post")},
    )
    c.pages[normalize_url(SOURCE)] = Page(
        url=SOURCE,
        title="Releases",
        text=f"v2.0 released Oct 5, 2026. {QUOTE} Older: v1.9 on Sep 1, 2026.",
        dates=["2026-09-01", "2026-10-05"],
    )
    c.learn([SOURCE])
    return c


@pytest.fixture
def tools(ctx):
    return {t.name: t for t in build_tools(ctx)}


def save(tools, **overrides):
    args = dict(
        title="AgentKit 2.0",
        url=f"{SOURCE}/tag/v2.0",
        source_page_url=SOURCE,
        published="2026-10-05",
        summary="AgentKit 2.0 adds parallel tool calls.",
        why_it_matters="Faster agents.",
        evidence_quote=QUOTE,
    )
    args.update(overrides)
    return tools["save_finding"].invoke(args)


def test_valid_finding_is_saved(tools, ctx):
    assert save(tools).startswith("Saved")
    assert ctx.findings[0].verified and ctx.findings[0].thread == "Frameworks"


def test_rejects_page_never_fetched(tools):
    assert "haven't fetched" in save(tools, source_page_url="https://made-up.example.com/news")


def test_rejects_invented_quote(tools):
    assert "copied exactly" in save(tools, evidence_quote="Adds 10x faster inference on every GPU.")


def test_quote_match_ignores_case_and_spacing(tools):
    assert save(tools, evidence_quote="adds   parallel tool calls\nto the agent runtime.").startswith("Saved")


def test_rejects_date_not_on_page(tools):
    assert "doesn't appear on that page" in save(tools, published="2026-10-06")


def test_rejects_unknown_date(tools):
    assert "doesn't appear on that page" in save(tools, published="unknown")


def test_rejects_news_older_than_window(tools):
    assert "before the window" in save(tools, published="2026-09-01")


def test_url_on_other_site_is_replaced_by_source(tools, ctx):
    save(tools, url="https://evil.example.com/fake")
    assert ctx.findings[0].url == SOURCE


def test_rejects_duplicate(tools):
    save(tools)
    assert save(tools) == "Already saved."


def test_memory_blocks_already_sent(tools):
    assert "Already posted" in tools["check_already_sent"].invoke({"url": "https://old.example.com/post/"})
    assert tools["check_already_sent"].invoke({"url": "https://new.example.com"}) == "Not posted before."


def test_fetch_page_blocks_guessed_urls(tools):
    # Checked before any network call: the URL was never seen in search results or on a page.
    assert "Don't guess URLs" in tools["fetch_page"].invoke({"url": "https://openai.com/blog/gpt-9"})


def test_page_dates_reads_iso_and_written_dates():
    html = '<relative-time datetime="2026-10-05T10:00:00Z">'
    text = "Shipped Oct 6, 2026. Also 7 October 2026 and Sept. 30, 2026."
    assert page_dates(html, text) == ["2026-09-30", "2026-10-05", "2026-10-06", "2026-10-07"]


class _Req:
    def __init__(self, args):
        self.tool_call = {"name": "web_search", "args": args, "id": "call-1"}


def _run(_req):
    return ToolMessage(content="ran", tool_call_id="call-1")


def test_loop_detector_blocks_repeats_then_stops_agent():
    ld = LoopDetector("t", max_repeats=2)
    results = [ld.wrap_tool_call(_Req({"query": q}), _run).content for q in ["a", "b", "a"]]
    assert results[:2] == ["ran", "ran"] and results[2].startswith("Loop detected")
    assert ld.before_model({}, None) is None  # one repeat: warn only

    ld.wrap_tool_call(_Req({"query": "a"}), _run)
    assert ld.before_model({}, None)["jump_to"] == "end"  # second repeat: stop the agent


def _billing_error():
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError("credit balance is too low", response=httpx.Response(400, request=req), body=None)


def test_search_outage_degrades_gracefully(tools, monkeypatch):
    from digest import costs, tools as tools_mod

    costs.start(1.0)
    calls = []

    def broken_search(*a, **k):
        calls.append(1)
        raise _billing_error()

    monkeypatch.setattr(tools_mod, "run_search", broken_search)
    first = tools["web_search"].invoke({"query": "langgraph release"})
    second = tools["web_search"].invoke({"query": "mcp release"})
    assert "unavailable" in first and "unavailable" in second
    assert len(calls) == 1  # after a billing error, search is switched off for the run


def test_writer_falls_back_to_free_model(monkeypatch):
    from langchain_core.messages import AIMessage

    from digest import nodes

    class Broken:
        def invoke(self, _):
            raise _billing_error()

    class Free:
        def invoke(self, _):
            return AIMessage(content=" digest from free model ")

    monkeypatch.setattr(nodes, "writer_chat", lambda: Broken())
    monkeypatch.setattr(nodes, "free_chat", lambda m: Free())
    out = nodes.writer({"selected": [], "today": "2026-10-07", "revisions": 0})
    assert out["digest"] == "digest from free model"
