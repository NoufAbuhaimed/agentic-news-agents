# Agentic News Agent

**Submitted by:** Nouf Abuhaimed · **Academy:** [@SDAIAAcademy](https://x.com/SDAIAAcademy)

A LangGraph multi-agent system that researches agentic-AI / LLM / dev-tools news and posts a
digest to a Signal group every other day. Architecture diagrams: [docs/architecture.html](docs/architecture.html).

```
planner ──► research agent ×4 (parallel, create_agent) ──► fact_check ──► select ──► writer ⇄ critic ──► send
                    ▲                                          │
                    └──── weak topic: retry once w/ feedback ──┘
```

| Node | Who does it | Job |
|---|---|---|
| planner | Free model | Split the brief into 4 topics with starting URLs; reads long-term source memory |
| research agent ×4 | `create_agent` · Nemotron (free) → Haiku fallback | Decides which tools to call and when to stop |
| ↳ `web_search` tool | Claude Haiku + Anthropic web search | ~2¢ per search, capped at 5 per agent |
| ↳ `fetch_page` tool | httpx + trafilatura | Free page reading |
| ↳ `check_already_sent` tool | SQLite memory | Skip news posted before |
| ↳ `save_finding` tool | Code | Rejects items unless the page was fetched and the evidence quote is exact |
| fact_check | Code + free model | Checks each summary against its quote; sends empty topics back for one retry |
| select | Code + free model | Dedupe, rank, keep top 7 |
| writer | Claude Sonnet 5.5 | Signal message under 2,000 characters |
| critic | Code + free model | Length/URL/claim checks; loops back to the writer at most twice |
| send | signal-cli | Post to the group, update memory |

**Guard rails:** model-call limit (30) and per-tool limits per agent, a loop detector (repeated identical tool
calls are blocked, and 3 repeats stop the agent), model fallback, node retries, and a **hard $1 budget per
run** (searches stop and the run wraps up). **Typical cost: ~$0.40-0.50 per run, about $6-8/month.**

## Requirements

**Accounts and keys**
- **Anthropic API key**, from console.anthropic.com. Pays for web search (Haiku) and the final writing (Sonnet).
- **OpenRouter API key**, from openrouter.ai. Free models do the planning, reading and checking.
- **Signal** on your phone. The agent links to your account as an extra device, like Signal Desktop.
- **LangSmith API key** (optional, free tier) for tracing, from smith.langchain.com.
- **healthchecks.io** check URL (optional, free) to get emailed when a run fails or is missed.

**Machine**
- An always-on Linux box with **Docker + Docker Compose**: a $5/mo VPS (Hetzner, DigitalOcean) or
  a Raspberry Pi 4/5. About 1 GB RAM is enough.
- For local testing, Python 3.11+ (the code uses 3.11 features).

**Python packages**: see `requirements.txt` (LangGraph, langchain-anthropic, langsmith, httpx,
pydantic, python-dotenv).

## Demo commands

```bash
.venv/bin/python -m digest graph          # architecture, built from the code: graph, agent, tools, guard rails
.venv/bin/python -m digest guardrails     # offline demo: invented quotes, wrong dates, guessed URLs, loops get caught
.venv/bin/python -m pytest -q             # 22 offline tests (no network, no cost)
.venv/bin/python -m digest run --dry-run  # a real run: live step log, then digest, retries, cost, LangSmith link
```

## 1. Try it locally (no Signal needed)

```bash
cd tech-digest
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # set ANTHROPIC_API_KEY (+ LANGSMITH_API_KEY, or LANGSMITH_TRACING=false)
.venv/bin/python -m digest run --dry-run
```

This runs the whole graph and prints the digest without sending or recording anything. Step logs
go to stdout and `data/logs/digest.log`. The full trace is in LangSmith under project `tech-digest`.

## 2. Deploy on the server

```bash
# copy the folder to the server (without .venv and data), then:
cp .env.example .env && nano .env          # ANTHROPIC_API_KEY, SIGNAL_NUMBER, LangSmith, TZ
docker compose up -d signal-api
```

**Link Signal.** On the server, open an SSH tunnel from your laptop (`ssh -L 8080:localhost:8080 you@server`),
then open `http://localhost:8080/v1/qrcodelink?device_name=tech-digest` in your browser.
On your phone, go to **Signal → Settings → Linked devices → +** and scan the code.

**Pick the group.** Add yourself to the target group first, then run:

```bash
docker compose run --rm worker python -m digest groups      # copy the id into SIGNAL_GROUP_ID
docker compose run --rm worker python -m digest test-send "digest bot connected ✅"
docker compose run --rm worker python -m digest run --dry-run
docker compose up -d worker                                  # starts the scheduler
```

## Operations

- **Schedule:** the worker checks daily at `RUN_HOUR` (in `TZ`). It runs only if the last digest
  is at least 47 hours old, which gives a steady every-other-day rhythm. Force a run with
  `docker compose run --rm worker python -m digest run --force`.
- **Failures:** each node retries 3 times with backoff. If the run still fails, it is marked failed,
  healthchecks gets a `/fail` ping, and the next run **resumes from the last LangGraph checkpoint**.
  Finished research is not redone.
- **Logs:** `docker compose logs -f worker`, or `data/logs/digest.log` (rotated, 5 × 5 MB).
  Each node logs start/finish, and each model call logs tokens and search/fetch counts.
- **Tracing:** LangSmith shows every node, prompt, search and cost per run. Prompts and fetched pages
  go to LangSmith's servers. Set `LANGSMITH_TRACING=false` to keep everything local.
- **Cost:** mostly web searches: 4 threads × `SEARCH_MAX_USES` (5) at about 1¢ each, plus Haiku tokens for
  the results. Lower `SEARCH_MAX_USES` to spend less, or raise it for broader coverage.
- **Back up** `signal-data/`. Losing it means re-linking the device. `data/` holds the
  sent-link history and checkpoints.
- **Refusal fallback** is on: if a model declines a request, the API retries it on a fallback model
  instead of failing the run.

## Layout

```
tests/             offline test suite
docs/              architecture diagrams
digest/
  __main__.py       CLI: run / schedule / graph / guardrails / groups / test-send
  demo.py           terminal demos and the run summary
  graph.py          main LangGraph graph (fan-out, fact-check loop, routing) + checkpointer
  nodes.py          planner, researcher, select, writer, critic, send
  prompts.py        all prompts (edit BRIEF to change the focus)
  state.py          graph state + structured output schemas
  agent.py          research agent (create_agent) + guard-rail middleware + loop detector
  tools.py          web_search, fetch_page, check_already_sent, save_finding
  costs.py          per-run cost tracking and the budget cap
  llm.py            Claude (search, writer) + free OpenRouter models with fallback
  fetch.py          free page download + text extraction
  store.py          sent-links + run history (SQLite)
  signal_client.py  signal-cli-rest-api client
  observability.py  logging, LangChain step logger, health pings
```
