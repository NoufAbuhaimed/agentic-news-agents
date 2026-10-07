"""Graph state and the structured shapes the LLM nodes return.

Two kinds of definitions live here:

1. Pydantic models (ResearchThread, Plan, Candidate, Selection, FactCheck, Critique). These describe
   the JSON we ask a model to return, so the code can VALIDATE the reply instead of trusting free text.
   The `description=` texts are shown to the model as part of the JSON schema.

2. DigestState: the shared "notebook" that every node in the graph reads from and writes to.
   A node returns only the keys it changed; LangGraph merges that into the state and saves a
   checkpoint after every node.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


# --- Shapes returned by models ----------------------------------------------------------------------

class ResearchThread(BaseModel):
    """One research topic, produced by the planner and handed to one research agent."""
    name: str = Field(description="Short label, e.g. 'Agent frameworks'")
    focus: str = Field(description="What this researcher should look for, in 1-3 sentences")
    # URLs the agent may open without searching (checked by code to really load; see planner()).
    starting_points: list[str] = Field(
        default_factory=list,
        description="Sites, repos or queries worth checking first",
    )


class Plan(BaseModel):
    """The planner's whole answer: the list of topics."""
    threads: list[ResearchThread] = Field(description="3 to 6 non-overlapping research threads")


class Candidate(BaseModel):
    """One news item found by a research agent (saved through the save_finding tool)."""
    title: str
    url: str = Field(description="Primary source URL (release notes, repo, official blog, paper)")
    published: str = Field(description="Publication date as YYYY-MM-DD, or 'unknown'")
    summary: str = Field(description="What happened, 1-2 sentences, factual")
    why_it_matters: str = Field(description="Why a builder of agentic AI systems should care, 1 sentence")
    evidence: str = Field(default="", description="Exact quote from the source page supporting the summary")
    # Set by code: the part of the source page around the quote, so fact_check can verify the whole summary.
    context: str = ""
    # Set by code, not the model: true when the item traces back to a page we actually fetched.
    verified: bool = False
    # Which topic (research agent) found it; used by the retry loop to count findings per topic.
    thread: str = ""


class CandidateList(BaseModel):
    items: list[Candidate]


class Selection(BaseModel):
    """The editor model's choice in select(): indices into the numbered candidate list."""
    chosen: list[int] = Field(description="Indices of the selected candidates, best first")
    reasoning: str = Field(description="One or two sentences on what was cut and why")


class Verdict(BaseModel):
    """fact_check's judgement on one candidate."""
    index: int
    supported: bool = Field(description="Is the summary fully supported by the evidence quote?")
    problem: str = Field(default="", description="If not supported, what is wrong")


class FactCheck(BaseModel):
    verdicts: list[Verdict]


class Critique(BaseModel):
    """The critic's review of the written digest; problems are sent back to the writer."""
    passed: bool
    problems: list[str] = Field(default_factory=list, description="Concrete fixes the writer must make")


# --- The shared state -------------------------------------------------------------------------------

def _merge(left: dict | None, right: dict | None) -> dict:
    """Reducer for dict fields: combine the old and new dicts instead of replacing the old one."""
    return {**(left or {}), **(right or {})}


class DigestState(TypedDict, total=False):
    # Inputs, set once by __main__.run() at the start.
    run_id: str
    today: str
    since: str  # ISO date: only cover news on/after this
    recent_urls: list[str]  # already-sent, normalized
    dry_run: bool

    # Written by the nodes as the run progresses.
    threads: list[ResearchThread]  # planner
    # `Annotated[..., operator.add]` is a REDUCER: 4 agents run in parallel and each returns its
    # own list; without the reducer the last one would overwrite the others, with it the lists are
    # concatenated. Same idea for the retry round: round 2 results are added to round 1.
    candidates: Annotated[list[Candidate], operator.add]  # fan-in from parallel research agents
    attempts: Annotated[dict[str, int], _merge]  # research attempts per topic
    checked: Annotated[list[Candidate], operator.add]  # candidates that passed fact_check
    fact_checked: Annotated[list[str], operator.add]  # keys of candidates already fact-checked
    fact_notes: Annotated[dict[str, list[str]], _merge]  # fact-check problems per topic, fed back on retry
    selected: list[Candidate]  # select (no reducer: only one node writes it, so replace is fine)

    digest: str  # writer: the Signal message
    critique: Critique | None  # critic
    revisions: int  # how many drafts the writer has produced (caps the quality loop)
    sent: bool  # send


class ResearcherInput(TypedDict, total=False):
    """The private input of ONE research agent, delivered by a `Send` (not the whole state)."""
    thread: ResearchThread
    today: str
    since: str
    recent_urls: list[str]
    attempt: int  # 1 = first try, 2 = retry sent back by fact_check
    feedback: str  # on a retry: why the first attempt found nothing usable
