"""Terminal demos: `digest graph`, `digest guardrails`, and the summary printed after `digest run`."""

from __future__ import annotations

import os

from .config import settings


def _h(title: str) -> None:
    print(f"\n{'═' * 70}\n {title}\n{'═' * 70}")


def show_graph() -> None:
    from .agent import build_research_agent
    from .graph import build_graph
    from .state import ResearchThread
    from .tools import ResearchContext, build_tools

    _h("MAIN GRAPH (LangGraph), built from the code")
    g = build_graph().get_graph()
    print("Nodes:", ", ".join(n for n in g.nodes if not n.startswith("__")))
    for e in g.edges:
        print(f"  {e.source:>10} → {e.target:<10} {'(conditional)' if e.conditional else ''}")
    print(
        "\nPatterns: planner decomposes the query · parallel research agents (Send) ·\n"
        "fact_check retry loop back to the researcher · writer ⇄ critic loop · conditional routing"
    )

    _h("RESEARCH AGENT (create_agent, a LangGraph subgraph)")
    ctx = ResearchContext(ResearchThread(name="demo", focus="demo"), "2026-01-02", "2026-01-01", set())
    agent = build_research_agent(ctx, settings.search_max_uses)
    print("Nodes:", ", ".join(n for n in agent.get_graph().nodes if not n.startswith("__")))
    print("\nTools the model chooses between:")
    for t in build_tools(ctx):
        print(f"  🔧 {t.name}: {t.description.splitlines()[0]}")
    print(
        f"\nModels: {' → '.join(settings.free_models[:2])} → {settings.search_model} (fallback)\n"
        f"Guard rails: {settings.agent_max_steps} model turns · {settings.search_max_uses} searches "
        f"({settings.retry_search_max_uses} on retry) · {settings.pages_per_thread} fetches · loop detector · "
        f"${settings.run_budget_usd:.2f} budget per run"
    )


def show_guardrails() -> None:
    import logging

    from langchain_core.messages import ToolMessage

    logging.getLogger("digest.agent").setLevel(logging.ERROR)  # the demo prints the outcomes itself
    from .agent import LoopDetector
    from .fetch import Page
    from .state import ResearchThread
    from .store import normalize_url
    from .tools import ResearchContext, build_tools

    _h("GUARD RAILS: what happens when an agent cuts corners (offline, free)")
    src = "https://github.com/acme/agentkit/releases"
    ctx = ResearchContext(ResearchThread(name="demo", focus="f"), "2026-10-07", "2026-10-04",
                          already_sent={normalize_url("https://old.example.com/post")})
    ctx.pages[normalize_url(src)] = Page(src, "Releases",
                                         "v2.0 released Oct 5, 2026. Adds parallel tool calls to the agent runtime.",
                                         dates=["2026-10-05"])
    ctx.learn([src])
    tools = {t.name: t for t in build_tools(ctx)}
    print(f"The agent has fetched one page: {src}\n")

    good = dict(title="AgentKit 2.0", url=src, source_page_url=src, published="2026-10-05",
                summary="Adds parallel tool calls.", why_it_matters="Faster agents.",
                evidence_quote="Adds parallel tool calls to the agent runtime.")
    cases = [
        ("cites a page it never read", {**good, "source_page_url": "https://made-up.example.com/news"}),
        ("invents a quote", {**good, "evidence_quote": "Makes inference 10x faster on every GPU."}),
        ("claims a date not on the page", {**good, "published": "2026-10-06"}),
        ("saves a valid finding", good),
        ("saves it again", good),
    ]
    for label, args in cases:
        print(f"  save_finding: {label:32} → {tools['save_finding'].invoke(args)[:90]}")
    print(f"  fetch_page: {'guesses a URL from memory':34} → {tools['fetch_page'].invoke({'url': 'https://openai.com/blog/gpt-9'})[:90]}")
    print(f"  check_already_sent: {'old link':26} → {tools['check_already_sent'].invoke({'url': 'https://old.example.com/post'})}")

    print("\nLoop detector:")

    class Req:
        def __init__(self, q):
            self.tool_call = {"name": "web_search", "args": {"query": q}, "id": "1"}

    ld = LoopDetector("demo", max_repeats=2)
    ran = lambda req: ToolMessage(content="(search ran)", tool_call_id="1")  # noqa: E731
    for q in ["mcp release", "langgraph release", "mcp release", "mcp release"]:
        print(f"  web_search({q!r:20}) → {ld.wrap_tool_call(Req(q), ran).content[:70]}")
    print(f"  next model turn → {'agent stopped' if ld.before_model({}, None) else 'continues'}")


def print_summary(out: dict, trace_run_id=None) -> None:
    result, cost = out["result"], out["cost"]

    _h("DIGEST")
    print(result.get("digest") or "(no new verified news in this window; nothing to send)")

    _h("WHAT HAPPENED")
    attempts = result.get("attempts", {})
    for t in result.get("threads", []):
        found = sum(1 for c in result.get("checked", []) if c.thread == t.name)
        n = attempts.get(t.name, 1)
        print(f"  🤖 {t.name}: {found} verified finding(s){f'  (retried: {n} attempts)' if n > 1 else ''}")
    for topic, notes in (result.get("fact_notes") or {}).items():
        for note in notes:
            print(f"  🔎 fact_check [{topic}]: {note}")
    print(f"  saved by agents: {len(result.get('candidates', []))} → passed fact_check: "
          f"{len(result.get('checked', []))} → selected: {len(result.get('selected', []))}")
    if result.get("critique") is not None:
        print(f"  🧐 critic: {'passed' if result['critique'].passed else 'sent with problems'} "
              f"after {result.get('revisions', 0)} draft(s)")

    _h("COST & TRACE")
    print(f"  {cost.summary()}  (budget ${settings.run_budget_usd:.2f})")
    if trace_run_id and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers
            from langsmith import Client

            wait_for_all_tracers()
            client = Client()
            print("  LangSmith trace:", client.get_run_url(run=client.read_run(trace_run_id),
                                                           project_name=os.getenv("LANGSMITH_PROJECT")))
        except Exception as e:  # tracing is optional; never fail the run over it
            print("  LangSmith trace link unavailable:", e.__class__.__name__)
    print(f"  Checkpoints + logs: {settings.checkpoint_path} · {settings.log_dir / 'digest.log'}")
