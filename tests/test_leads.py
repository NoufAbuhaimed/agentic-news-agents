"""Newsletter leads: parsing an issue, cleaning links, failing safely, and reaching the planner."""

from digest import fetch, nodes
from digest.state import Plan, ResearchThread

ISSUE = """
<a href="https://zenity.io/summit?utm_source=tldr">Agents in production (Sponsor)</a>
<a href="https://blog.google/technology/embeddinggemma-2/?utm_source=tldrai&amp;amp;utm_medium=x">
  <h3>EmbeddingGemma 2 (5 minute read)</h3></a>
<a href="https://mistral.ai/news/mistral-large-4/?utm_source=tldrai">Mistral Large 4 (9 minute read)</a>
<a href="https://github.com/acme/agent-kit?utm_source=tldrai">Agent Kit (GitHub Repo)</a>
<a href="https://www.collibra.com/learn?utm_source=mediadirect">Get your AI Trust Score.</a>
<a href="https://tldr.tech/ai/2026-10-06">Previous issue</a>
<a href="https://mistral.ai/news/mistral-large-4/?utm_source=tldrai">Mistral Large 4 (9 minute read)</a>
"""


def test_issue_stories_keeps_real_stories_only():
    stories = fetch.issue_stories(ISSUE)
    assert [s["headline"] for s in stories] == ["EmbeddingGemma 2", "Mistral Large 4", "Agent Kit"]
    assert stories[0]["url"] == "https://blog.google/technology/embeddinggemma-2/"  # tracking removed
    # sponsors, call-to-action links, newsletter-internal links and duplicates are dropped


def test_newsletter_failure_returns_no_leads(monkeypatch):
    def boom(*a, **k):
        raise fetch.httpx.ConnectError("offline")

    monkeypatch.setattr(fetch.httpx, "get", boom)
    assert fetch.newsletter_leads("2026-10-06") == []


def test_planner_passes_leads_to_the_model(monkeypatch):
    seen = {}

    def model(messages, schema):
        seen["prompt"] = messages[-1].content
        return Plan(threads=[ResearchThread(name="LLMs", focus="f",
                                            starting_points=["https://blog.google/technology/embeddinggemma-2/"])])

    monkeypatch.setattr(nodes, "newsletter_leads", lambda since: [
        {"headline": "EmbeddingGemma 2", "url": "https://blog.google/technology/embeddinggemma-2/", "issue_date": since}])
    monkeypatch.setattr(nodes, "free_json", model)
    monkeypatch.setattr(nodes, "reachable", lambda urls: urls)
    out = nodes.planner({"today": "2026-10-07", "since": "2026-10-06"})
    assert "EmbeddingGemma 2 → https://blog.google/technology/embeddinggemma-2/" in seen["prompt"]
    assert out["threads"][0].starting_points == ["https://blog.google/technology/embeddinggemma-2/"]
