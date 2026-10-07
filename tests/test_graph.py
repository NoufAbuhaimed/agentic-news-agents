"""Graph shape, routing (fact-check retry loop, writer ⇄ critic), memory, budget, and a full offline run."""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.types import Send

from digest import costs, nodes
from digest.config import settings
from digest.graph import build_graph
from digest.state import Candidate, Critique, ResearchThread
from digest.store import Store, normalize_url

T1, T2 = ResearchThread(name="A", focus="a"), ResearchThread(name="B", focus="b")


def cand(url, thread="A", title="t"):
    return Candidate(title=title, url=url, published="2026-10-05", summary="s", why_it_matters="w",
                     evidence="e", verified=True, thread=thread)


def base_state(**kw):
    s = {"today": "2026-10-07", "since": "2026-10-04", "threads": [T1, T2], "recent_urls": [],
         "checked": [], "attempts": {"A": 1, "B": 1}, "fact_notes": {}}
    s.update(kw)
    return s


def test_graph_has_parallel_fanout_retry_loop_and_routing():
    g = build_graph().get_graph()
    edges = {(e.source, e.target, e.conditional) for e in g.edges}
    assert ("planner", "researcher", True) in edges          # parallel fan-out (Send)
    assert ("researcher", "fact_check", False) in edges      # fan-in
    assert ("fact_check", "researcher", True) in edges       # fact-check retry loop
    assert ("fact_check", "select", True) in edges
    assert ("select", "__end__", True) in edges              # routing: nothing new → end
    assert ("critic", "writer", True) in edges               # quality loop


def test_fact_check_sends_only_empty_topics_back_with_feedback():
    state = base_state(checked=[cand("https://x.dev/1", thread="A")], fact_notes={"B": ["'x' was rejected: wrong date"]})
    route = nodes.after_fact_check(state)
    assert isinstance(route, list) and len(route) == 1
    send: Send = route[0]
    assert send.node == "researcher" and send.arg["thread"].name == "B" and send.arg["attempt"] == 2
    assert "wrong date" in send.arg["feedback"]


def test_fact_check_retries_each_topic_only_once():
    assert nodes.after_fact_check(base_state(attempts={"A": 2, "B": 2})) == "select"


def test_no_retries_once_budget_is_spent():
    costs.start(0.0)  # budget 0 → already over
    assert nodes.after_fact_check(base_state()) == "select"


def test_select_drops_already_sent_and_duplicates():
    state = base_state(
        checked=[cand("https://x.dev/1"), cand("https://x.dev/1/"), cand("https://x.dev/2")],
        recent_urls=[normalize_url("https://x.dev/2")],
    )
    assert [c.url for c in nodes.select(state)["selected"]] == ["https://x.dev/1"]


def test_critic_loop_is_bounded():
    bad = Critique(passed=False, problems=["fix"])
    assert nodes.after_critic({"critique": bad, "revisions": 1}) == "writer"
    assert nodes.after_critic({"critique": bad, "revisions": settings.max_revisions + 1}) == "send"


def test_long_term_memory_remembers_sources(tmp_path):
    db = Store(tmp_path / "m.sqlite")
    db.record_sent([("https://github.com/a/b/releases/tag/v1", "x"), ("https://www.github.com/c", "y"), ("https://blog.dev/p", "z")])
    assert db.top_sources()[0] == "github.com"
    assert normalize_url("https://github.com/a/b/releases/tag/v1") in db.recent_urls(14)


def test_cost_tracker_prices_claude_and_searches_and_free_models():
    tracker = costs.start(1.0)

    def result(model, tin, tout, searches=0):
        msg = AIMessage(content="", response_metadata={"model": model, "usage": {"server_tool_use": {"web_search_requests": searches}}},
                        usage_metadata={"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout})
        return LLMResult(generations=[[ChatGeneration(message=msg)]])

    tracker.on_llm_end(result("claude-haiku-4-5-20251001", 10_000, 200, searches=1))  # 0.01 + 0.001 + 0.01
    tracker.on_llm_end(result("nvidia/nemotron-3-super-120b-a12b:free", 50_000, 5_000))
    assert abs(tracker.total_usd - 0.021) < 1e-9
    assert not costs.over_budget()
    tracker.on_llm_end(result("claude-sonnet-5-5", 0, 100_000))  # +$1.00
    assert costs.over_budget()


def test_full_run_offline_with_retry(monkeypatch):
    """planner → 2 agents (B finds nothing) → fact_check retries B → select → writer → critic → END."""
    calls = {"research": []}

    def fake_free_json(messages, schema):
        name = schema.__name__
        if name == "Plan":
            return schema(threads=[T1, T2])
        if name == "FactCheck":
            return schema(verdicts=[])
        if name == "Critique":
            return schema(passed=True)
        raise AssertionError(name)

    def fake_researcher(inp):
        calls["research"].append((inp["thread"].name, inp.get("attempt", 1)))
        found = [] if (inp["thread"].name == "B" and inp.get("attempt", 1) == 1) else [cand(f"https://x.dev/{inp['thread'].name}", thread=inp["thread"].name)]
        return {"candidates": found, "attempts": {inp["thread"].name: inp.get("attempt", 1)}}

    class FakeWriter:
        def invoke(self, messages):
            return AIMessage(content="digest https://x.dev/A https://x.dev/B", response_metadata={"stop_reason": "end_turn"})

    monkeypatch.setattr(nodes, "free_json", fake_free_json)
    monkeypatch.setattr(nodes, "researcher", fake_researcher)
    monkeypatch.setattr(nodes, "writer_chat", lambda: FakeWriter())

    out = build_graph().invoke(
        {"today": "2026-10-07", "since": "2026-10-04", "recent_urls": [], "dry_run": True,
         "candidates": [], "checked": [], "fact_checked": [], "attempts": {}, "fact_notes": {}, "revisions": 0},
        {"configurable": {"thread_id": "test"}},
    )
    assert sorted(calls["research"]) == [("A", 1), ("B", 1), ("B", 2)]  # B was retried once
    assert {c.url for c in out["selected"]} == {"https://x.dev/A", "https://x.dev/B"}
    assert out["digest"].startswith("digest") and out["sent"] is False


def test_empty_run_does_not_move_the_window(tmp_path):
    db = Store(tmp_path / "r.sqlite")
    db.start_run("r1")
    db.finish_run("r1", "empty", 0.5)
    assert db.last_sent_at() is None  # nothing was posted, so the next run keeps the full window
    db.start_run("r2")
    db.finish_run("r2", "sent", 0.4)
    assert db.last_sent_at() is not None
