"""Graph state and the structured shapes the LLM nodes return."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchThread(BaseModel):
    name: str = Field(description="Short label, e.g. 'Agent frameworks'")
    focus: str = Field(description="What this researcher should look for, in 1-3 sentences")
    starting_points: list[str] = Field(
        default_factory=list,
        description="Sites, repos or queries worth checking first",
    )


class Plan(BaseModel):
    threads: list[ResearchThread] = Field(description="3 to 6 non-overlapping research threads")


class Candidate(BaseModel):
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
    thread: str = ""


class CandidateList(BaseModel):
    items: list[Candidate]


class Selection(BaseModel):
    chosen: list[int] = Field(description="Indices of the selected candidates, best first")
    reasoning: str = Field(description="One or two sentences on what was cut and why")


class Verdict(BaseModel):
    index: int
    supported: bool = Field(description="Is the summary fully supported by the evidence quote?")
    problem: str = Field(default="", description="If not supported, what is wrong")


class FactCheck(BaseModel):
    verdicts: list[Verdict]


class Critique(BaseModel):
    passed: bool
    problems: list[str] = Field(default_factory=list, description="Concrete fixes the writer must make")


def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class DigestState(TypedDict, total=False):
    run_id: str
    today: str
    since: str  # ISO date: only cover news on/after this
    recent_urls: list[str]  # already-sent, normalized
    dry_run: bool

    threads: list[ResearchThread]
    candidates: Annotated[list[Candidate], operator.add]  # fan-in from parallel research agents
    attempts: Annotated[dict[str, int], _merge]  # research attempts per topic
    checked: Annotated[list[Candidate], operator.add]  # candidates that passed fact_check
    fact_checked: Annotated[list[str], operator.add]  # keys of candidates already fact-checked
    fact_notes: Annotated[dict[str, list[str]], _merge]  # fact-check problems per topic, fed back on retry
    selected: list[Candidate]

    digest: str
    critique: Critique | None
    revisions: int
    sent: bool


class ResearcherInput(TypedDict, total=False):
    thread: ResearchThread
    today: str
    since: str
    recent_urls: list[str]
    attempt: int
    feedback: str
