# Code guide: how Agentic News Agent works

This guide follows **one run** from the moment it starts until the digest lands in Signal, file by file.
Read it top to bottom with the code open next to it. Every excerpt is from the real code.

**Contents**
1. [The 30-second version](#1-the-30-second-version)
2. [Three ideas you need first](#2-three-ideas-you-need-first)
3. [The files at a glance](#3-the-files-at-a-glance)
4. [Following one run](#4-following-one-run) — start → planner → agents → fact_check → select → writer ⇄ critic → send
5. [Inside a research agent](#5-inside-a-research-agent)
6. [The supporting files](#6-the-supporting-files)
7. [Reliability: what happens when things go wrong](#7-reliability-what-happens-when-things-go-wrong)
8. [Tests](#8-tests)
9. [How to change common things](#9-how-to-change-common-things)
10. [Check your understanding](#10-check-your-understanding)

---

## 1. The 30-second version

Every other day the program:

1. **Plans**: a model splits "agentic-AI news" into 4 topics.
2. **Researches**: 4 AI agents work **at the same time**, one per topic. Each one searches the web, reads pages,
   and saves news items, but only items it can prove with an exact quote and a date from a page it really read.
3. **Fact-checks**: another model checks each item against its source. Topics with nothing verified get **one
   retry** with feedback.
4. **Selects**: removes anything posted before and picks the best items.
5. **Writes**: Claude writes the Signal message; a critic checks it and can send it back for fixes.
6. **Sends**: posts it to the Signal group and remembers what was posted.

---

## 2. Three ideas you need first

### 2.1 A graph (LangGraph)

The program is a **graph**: boxes (**nodes**) connected by arrows (**edges**). Each node is a normal Python
function. Some arrows are fixed; others are **conditional**: a small function looks at the data and decides
where to go next. That is how we get loops (retry, rewrite) and early exits (nothing new → stop).

### 2.2 Shared state

All nodes read and write one shared dictionary, the **state** (defined in `state.py`). A node receives the
state and **returns only the keys it wants to change**. LangGraph merges that into the state, and after every
node it saves a **checkpoint** (a snapshot) to SQLite.

### 2.3 An agent (`create_agent`)

A normal function does fixed steps. An **agent** is a loop where the *model* decides the steps:

```
model: "what should I do next?" → calls a tool → reads the result → decides again → ... → answers with no tool call = done
```

We give it **tools** (Python functions) and **guard rails** (limits that stop it looping or overspending).

---

## 3. The files at a glance

```
digest/
├── __main__.py       START HERE: the commands (run, schedule, graph, guardrails, …)
├── graph.py          the map: which nodes exist and how they connect
├── state.py          the shape of the shared data
├── nodes.py          the code of every node (planner, researcher, fact_check, select, writer, critic, send)
├── agent.py          builds one research agent + its guard rails
├── tools.py          the 4 tools the agent can use
├── prompts.py        every instruction given to a model, in plain English
├── llm.py            connections to Claude and to the free OpenRouter models
├── fetch.py          downloads web pages and extracts their text (no AI)
├── store.py          long-term memory in SQLite
├── costs.py          adds up spending, enforces the $1 budget
├── observability.py  logging + LangSmith helpers
├── signal_client.py  talks to the Signal container
├── config.py         reads settings from .env
└── demo.py           the `graph` / `guardrails` demos and the end-of-run summary
```

**Suggested reading order:** `graph.py` → `state.py` → `nodes.py` → `agent.py` → `tools.py` → the rest.

---

## 4. Following one run

### Step 0 · Starting: `__main__.py`

You type `python -m digest run --dry-run`. `main()` reads the command and calls `run()`, which:

1. **Checks the timer.** If a digest was posted less than 47 hours ago, it stops (unless `--force` or `--dry-run`).
2. **Builds the graph** with `build_graph()`.
3. **Resumes or starts.** If the last run crashed, it continues from its checkpoint instead of starting over:
   ```python
   resume_id = None if dry_run else db.unfinished_run()
   ...
   if resume_id:
       result = execute(None)          # None = "continue from the saved checkpoint"
   ```
4. **Starts a new run** with the initial state: today's date, the date window (`since`), links already posted
   (`recent_urls`), and empty lists for everything else.
5. **Records the result**: `"sent"` if something was posted, `"empty"` if there was no news, `"failed"` on a crash.

It also creates the **cost tracker** (`costs.start(...)`) and passes logging callbacks so every model call is
logged and priced.

### Step 1 · The map: `graph.py`

```python
g.add_node("planner", nodes.planner, retry_policy=_RETRY)
g.add_node("researcher", nodes.researcher, retry_policy=_RETRY)
...
g.add_edge(START, "planner")
g.add_conditional_edges("planner", nodes.fan_out, ["researcher"])                       # → 4 agents in parallel
g.add_edge("researcher", "fact_check")                                                # waits for all 4
g.add_conditional_edges("fact_check", nodes.after_fact_check, ["researcher", "select"]) # retry loop
g.add_conditional_edges("select", nodes.after_select, {"writer": "writer", "end": END})
g.add_edge("writer", "critic")
g.add_conditional_edges("critic", nodes.after_critic, {"writer": "writer", "send": "send"})  # quality loop
g.add_edge("send", END)
```

- `add_edge` = a fixed arrow. `add_conditional_edges` = a function decides.
- `retry_policy=_RETRY` = if a node fails because of a network problem or rate limit, LangGraph retries it up to
  3 times (but not for errors that can't fix themselves, like "no credits").
- `SqliteSaver` = the **checkpointer**: it saves the state after every node to `data/checkpoints.sqlite`.

### Step 2 · The shared data: `state.py`

```python
class DigestState(TypedDict, total=False):
    today: str; since: str; recent_urls: list[str]; dry_run: bool
    threads: list[ResearchThread]                              # from the planner
    candidates: Annotated[list[Candidate], operator.add]       # from the agents
    attempts: Annotated[dict[str, int], _merge]                # research tries per topic
    checked: Annotated[list[Candidate], operator.add]          # passed fact_check
    fact_notes: Annotated[dict[str, list[str]], _merge]        # why items were rejected
    selected: list[Candidate]                                  # chosen by select
    digest: str; critique: Critique | None; revisions: int; sent: bool
```

`Annotated[..., operator.add]` is a **reducer**. Four agents run at once and each returns its own `candidates`
list. Without the reducer the last one would overwrite the others; with it, the lists are **added together**.

`Candidate` is one news item: `title, url, published, summary, why_it_matters, evidence, context, verified, thread`.

The other classes (`Plan`, `Selection`, `FactCheck`, `Critique`) describe the **JSON we ask models to return**,
so the code can check the reply has the right shape.

### Step 3 · Planner: `nodes.planner`

```python
plan = free_json([SystemMessage(prompts.PLANNER), HumanMessage(f"Today is ... {memory}")], Plan)
threads = plan.threads[:4]
for t in threads:
    t.starting_points = reachable(t.starting_points)   # drop URLs that don't load
return {"threads": threads}
```

- A **free model** returns a `Plan`: 4 topics, each with a focus and starting URLs.
- `memory` adds the sources that produced published items before (long-term memory, from `store.top_sources()`).
- `reachable()` (in `fetch.py`) tests each URL, because models sometimes invent URLs.

### Step 4 · Fan-out: `nodes.fan_out`

```python
return [_send_research(state, t) for t in state["threads"]]
```

Each `Send("researcher", {...})` starts one copy of the `researcher` node with its own input. Four `Send`s =
**four agents running in parallel**. This is the "parallel branches" pattern.

### Step 5 · Research agent: `nodes.researcher`

```python
ctx = ResearchContext(thread=thread, today=..., since=..., already_sent=...)
ctx.learn(list(thread.starting_points) + prompts.SEED_URLS)       # URLs the agent may open
task = "Today is ... Your topic: ... Starting URLs: ..."           # the agent's instructions
if inp.get("feedback"):
    task += "This is retry #1. Feedback from the fact-checker: ..."
agent = build_research_agent(ctx, search_limit, max_steps, time_limit)
agent.invoke({"messages": [HumanMessage(task)]}, ...)
return {"candidates": ctx.findings, "attempts": {thread.name: attempt}}
```

- `ResearchContext` is the agent's **notebook**: pages it fetched, URLs it's allowed to open, findings it saved.
- A retry gets smaller limits (3 searches, 12 turns, 2 minutes) so it can't double the cost.
- If the agent crashes **after** saving findings, we keep them; if it crashes with nothing, the error goes up so
  the retry policy / checkpoint can handle it.

What happens *inside* `agent.invoke` is explained in [section 5](#5-inside-a-research-agent).

### Step 6 · Fact-check and the retry loop: `nodes.fact_check` + `nodes.after_fact_check`

`fact_check` takes the **new** candidates and asks a free model: "is this summary supported by this excerpt
of its source page?" Supported items go to `checked`; rejected ones get a note in `fact_notes`.

Then `after_fact_check` decides where to go:

```python
if costs.over_budget():                       return "select"   # no money for retries
if len(state["checked"]) >= settings.enough_items:  return "select"   # already enough news
for each topic with 0 verified items and only 1 attempt so far:
    retries.append(_send_research(state, t, attempt + 1, feedback))
return retries or "select"
```

Returning a **list of `Send`s** sends those topics **back** to `researcher`, which is the **fact-check retry loop**.
After the retries finish, the graph comes back to `fact_check`, which only checks the new items.

### Step 7 · Select: `nodes.select` + `nodes.after_select`

1. **In code**: drops unverified items, duplicates, and anything in `recent_urls` (already posted).
2. **Editor model** (free) picks up to 7, best first, and may drop weak items (tutorials, opinion…).
3. `after_select`: if nothing is left → **END** (no empty message is sent). Otherwise → `writer`.

### Step 8 · Writer ⇄ critic: the quality loop

- `writer`: **Claude Sonnet** writes the Signal message from the selected items (format in `prompts.WRITER`).
  If Claude is unavailable, a free model writes it instead.
- `critic`: first **code checks** (length ≤ 2000 characters, every URL must belong to an item), then a free model
  checks the claims.
- `after_critic`: problems and fewer than 3 drafts → back to `writer` with the problems listed; otherwise → `send`.

### Step 9 · Send: `nodes.send`

```python
if state.get("dry_run"): return {"sent": False}
signal_client.send(state["digest"])
store().record_sent([(c.url, c.title) for c in state["selected"]])
```

Posts the message, then saves the links to memory so they're never posted twice. Back in `__main__.py`,
`print_summary()` (in `demo.py`) shows the digest, retries, cost, and the LangSmith link.

---

## 5. Inside a research agent

### 5.1 Building it: `agent.py`

```python
create_agent(
    free_chat(models[0]),                 # the "brain": Nemotron (free)
    build_tools(ctx),                     # the 4 tools
    system_prompt=prompts.AGENT,          # how to work (read it in prompts.py)
    middleware=[
        ModelFallbackMiddleware(...),     # brain fails → second free model → Claude Haiku
        ModelCallLimitMiddleware(...),    # max 20 turns (12 on retry)
        ToolCallLimitMiddleware("web_search", ...),   # max 5 searches (3 on retry)
        ToolCallLimitMiddleware("fetch_page", ...),   # max 8 page reads
        LoopDetector(...),                # ours: blocks repeated identical calls
        TimeLimit(...),                   # ours: 4 minutes (2 on retry)
    ],
)
```

**Middleware** = code that runs around every model turn or tool call. Two are ours:

- `LoopDetector.wrap_tool_call` runs **before each tool call**. If the exact same tool + arguments was already
  called, it returns a warning instead of running it; after 3 repeats, `before_model` ends the agent
  (`{"jump_to": "end"}`).
- `TimeLimit.before_model` runs **before each model turn**, and ends the agent once its time is up. Findings
  already saved are kept.

### 5.2 The tools: `tools.py`

The model sees each tool's **name and docstring** and decides when to call it.

| Tool | What it does | Key rule |
|---|---|---|
| `web_search(query)` | Runs one Anthropic web search through Claude Haiku, returns titles + URLs | Stops when the run's budget is used; if search is down, tells the agent to continue with page reading |
| `fetch_page(url)` | Downloads a page (via `fetch.py`), returns its text | Only URLs the agent has **seen** (start list, search results, links on pages it read): no guessing |
| `check_already_sent(url)` | Looks the URL up in memory | |
| `save_finding(...)` | Saves one news item | Rejected unless all checks pass ↓ |

`save_finding` is the heart of the "no hallucinations" design. It rejects the item if:

1. the `source_page_url` was **never fetched**;
2. the `evidence_quote` is **not on that page** (exact text, ignoring spacing/case);
3. the `published` date **doesn't appear on the page**, or is **before the window**;
4. it was **already posted** or **already saved**.

It also replaces a URL that points to a different site than the page it read. When the item passes, it stores
the quote plus ~5,000 characters around it (`context`), which `fact_check` uses later.

---

## 6. The supporting files

**`prompts.py`**: all model instructions. `BRIEF` defines what counts as news; `SEED_URLS` are fixed starting
sources; `PLANNER`, `AGENT`, `FACT_CHECK`, `SELECT`, `WRITER`, `CRITIC` are the per-step instructions.

**`llm.py`**: model connections.
- `free_chat(model)`: a free OpenRouter model (via `ChatOpenAI` with OpenRouter's URL).
- `free_json(messages, Schema)`: asks free models for JSON, **validates** it against the schema, and tries the
  next model if the reply is broken or the model is down.
- `run_search` / `search_results`: Claude Haiku with Anthropic's web search tool; extracts the result links.
- `writer_chat()`: Claude Sonnet for writing.

**`fetch.py`**: no AI. `fetch_pages` downloads pages in parallel and uses `trafilatura` to extract the article
text, plus all links (`links`) and all dates (`dates`) on the page, which `save_finding` uses to verify. `reachable`
checks which URLs load.

**`store.py`**: SQLite memory (`data/digest.sqlite`): `sent_items` (posted links), `source_stats` (which sites
produced published news), `runs` (history, status, cost).

**`costs.py`**: `CostTracker` is a LangChain callback: after **every** model call it reads the token counts and
search count, multiplies by the price table, and adds it up. `over_budget()` is checked by `web_search` and by the
retry decision. `disable_search()` switches search off for the rest of a run after a billing/auth error.

**`observability.py`**: `StepLogger` writes one log line per node, model call and tool call (the `🔧` lines).
LangSmith tracing needs no code: it turns on from `.env` (`LANGSMITH_TRACING=true`).

**`config.py`**: every setting with its default, overridable in `.env` (models, limits, budget, schedule, Signal).

**`signal_client.py`**: two HTTP calls to the Signal container: list groups, send a message.

---

## 7. Reliability: what happens when things go wrong

| Problem | What handles it | Where |
|---|---|---|
| Network error / rate limit in a node | Retried up to 3× with backoff | `graph.py` `_RETRY` |
| The whole run crashes | Next run resumes from the last checkpoint | `__main__.py` `unfinished_run()` |
| Free model busy or removed | Next free model, then Claude Haiku | `agent.py` fallback, `llm.free_json` |
| Out of Anthropic credits | Search switches off, agents keep reading pages; writer uses a free model | `tools.web_search`, `nodes.writer` |
| Agent invents a URL / quote / date | Rejected by the tools | `tools.fetch_page`, `tools.save_finding` |
| Agent loops or takes too long | `LoopDetector`, `TimeLimit`, call limits | `agent.py` |
| Spending too high | $1 cap stops searches and retries | `costs.py` |
| No new news | Run ends without sending; recorded as "empty" | `after_select`, `__main__.py` |

---

## 8. Tests

`tests/` has 28 tests that run offline in ~2 seconds (`python -m pytest -q`). They replace models with fakes,
so each test shows one behaviour clearly. Good ones to read:

- `test_full_run_offline_with_retry`: the whole graph, where one topic finds nothing and is retried once.
- `test_rejects_invented_quote`, `test_rejects_date_not_on_page`: the `save_finding` rules.
- `test_loop_detector_blocks_repeats_then_stops_agent`, `test_time_limit_stops_agent`: the guard rails.
- `test_search_outage_degrades_gracefully`: what happens when credits run out.

---

## 9. How to change common things

| I want to… | Change |
|---|---|
| Cover different topics | `BRIEF` in `prompts.py` |
| Add/remove news sources | `SEED_URLS` in `prompts.py` |
| Change the message format | `WRITER` in `prompts.py` |
| Spend less / search more | `SEARCH_MAX_USES`, `RUN_BUDGET_USD` in `.env` |
| Make runs faster | `AGENT_TIME_LIMIT_S`, `AGENT_MAX_STEPS`, `ENOUGH_ITEMS` in `.env` |
| Use different free models | `FREE_MODELS` in `.env` |
| Change the schedule | `RUN_HOUR`, `MIN_HOURS_BETWEEN`, `TZ` in `.env` |
| Add a new tool for the agent | a new `@tool` function in `build_tools()` in `tools.py`, then mention it in `prompts.AGENT` |
| Add a new step to the graph | a function in `nodes.py` + `add_node` / edges in `graph.py` |

---

## 10. Check your understanding

Try to answer, then check the file in brackets.

1. How do 4 agents run at the same time? *(`fan_out` returns 4 `Send`s, in `nodes.py`)*
2. Why don't the agents' findings overwrite each other? *(the `operator.add` reducer, in `state.py`)*
3. What makes the graph go back to research? *(`after_fact_check` returning `Send`s)*
4. How does an agent decide it's finished? *(it answers without calling a tool; or a guard rail stops it)*
5. Where would an invented quote be caught? *(`save_finding`, in `tools.py`)*
6. What happens if the program crashes halfway? *(checkpoint in `graph.py` + `unfinished_run()` in `__main__.py`)*
7. Where is the cost added up, and who reads it? *(`CostTracker` in `costs.py`; read by `web_search` and `after_fact_check`)*

**Try it:** run `python -m digest graph` and `python -m digest guardrails`, then a dry run, and match each log line
to the code that printed it.
