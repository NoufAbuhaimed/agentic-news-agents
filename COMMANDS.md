# Tech Digest: commands

Run these from the `tech-digest/` folder.

## One-time setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env    # then fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, LANGSMITH_API_KEY
```

## Demo (for presenting the project)

| # | Command | What it shows | Cost / time |
|---|---|---|---|
| 1 | `.venv/bin/python -m digest graph` | The architecture, built from the code: main graph nodes and edges, the research agent, its tools and guard rails | free · instant |
| 2 | `.venv/bin/python -m digest guardrails` | The guard rails catching an agent that invents quotes, claims wrong dates, guesses URLs or loops | free · instant |
| 3 | `.venv/bin/python -m pytest -q` | 22 automated tests: guard rails, routing, retry loop, memory, cost tracking | free · ~2 s |
| 4 | `.venv/bin/python -m digest run --dry-run` | **A real run on today's news.** Live log of every step and tool call, then the digest, retries, cost and the LangSmith trace link. Nothing is posted | ~$0.40–0.80 · 5–8 min |

```bash
.venv/bin/python -m digest graph
.venv/bin/python -m digest guardrails
.venv/bin/python -m pytest -q
.venv/bin/python -m digest run --dry-run
```

In the run log, look for:
- `planned 4 threads` → the planner splitting the query
- `🔧 web_search(...)`, `🔧 fetch_page(...)`, `🔧 save_finding(...)` → the agents choosing tools
- `loop detector[...]` → a repeated call caught
- `fact_check: sending N topic(s) back to research` → the retry loop
- `critic: pass` → the quality loop
- `run cost: $...` → cost per model, against the $1 budget

The full trace (every node, agent step, tool call and token count) is in LangSmith under the `tech-digest` project.

## Signal

Signal runs in Docker (`docker compose up -d signal-api`). Docker Desktop must be running.

```bash
.venv/bin/python -m digest groups                       # list your groups and their IDs
.venv/bin/python -m digest test-send "hello from the bot"  # send a test message to SIGNAL_GROUP_ID
.venv/bin/python -m digest run --force                  # a real run that POSTS the digest to the group
```

Messages are sent from **your own** linked account, so they appear in the group as your messages.

## Running it automatically

```bash
.venv/bin/python -m digest schedule    # checks daily at RUN_HOUR, posts every other day; keep it running
docker compose up -d                   # or run the scheduler and Signal together in Docker
```

## Useful checks

```bash
tail -f data/logs/digest.log                                        # live log
sqlite3 data/digest.sqlite "select run_id, status, cost_usd from runs order by started_at desc limit 5"
sqlite3 data/digest.sqlite "select domain, published from source_stats order by published desc"
```
