"""Runtime settings, read once from the environment (.env in dev, compose env in prod)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value if value not in ("", None) else default


@dataclass(frozen=True)
class Settings:
    # Models. Claude Haiku runs Anthropic web search; Claude Sonnet writes the final message;
    # free OpenRouter models (tried in order) plan, read pages, rank and review.
    search_model: str = field(default_factory=lambda: _env("SEARCH_MODEL", "claude-haiku-4-5"))
    writer_model: str = field(default_factory=lambda: _env("WRITER_MODEL", "claude-sonnet-5-5"))
    free_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip()
            for m in _env(
                "FREE_MODELS",
                "nvidia/nemotron-3-super-120b-a12b:free,"
                "nvidia/nemotron-3-ultra-550b-a55b:free,"
                "google/gemma-4-31b-it:free",
            ).split(",")
            if m.strip()
        )
    )

    # Research agent guard rails (per agent run). Searches are the main cost (~1.7c each).
    search_max_uses: int = field(default_factory=lambda: int(_env("SEARCH_MAX_USES", "5")))
    pages_per_thread: int = field(default_factory=lambda: int(_env("PAGES_PER_THREAD", "8")))
    retry_search_max_uses: int = field(default_factory=lambda: int(_env("RETRY_SEARCH_MAX_USES", "3")))
    retry_max_steps: int = field(default_factory=lambda: int(_env("RETRY_MAX_STEPS", "12")))
    agent_max_steps: int = field(default_factory=lambda: int(_env("AGENT_MAX_STEPS", "20")))
    page_max_chars: int = field(default_factory=lambda: int(_env("PAGE_MAX_CHARS", "12000")))  # kept for verification
    agent_page_chars: int = field(default_factory=lambda: int(_env("AGENT_PAGE_CHARS", "6000")))  # shown to the agent
    agent_time_limit_s: int = field(default_factory=lambda: int(_env("AGENT_TIME_LIMIT_S", "240")))
    retry_time_limit_s: int = field(default_factory=lambda: int(_env("RETRY_TIME_LIMIT_S", "120")))
    enough_items: int = field(default_factory=lambda: int(_env("ENOUGH_ITEMS", "3")))  # skip retries at/above this
    max_research_attempts: int = field(default_factory=lambda: int(_env("MAX_RESEARCH_ATTEMPTS", "2")))

    # Hard spending cap per run (USD). Once reached, agents stop searching and the run wraps up.
    run_budget_usd: float = field(default_factory=lambda: float(_env("RUN_BUDGET_USD", "1.00")))

    # Digest shape.
    max_items: int = field(default_factory=lambda: int(_env("MAX_ITEMS", "7")))
    max_chars: int = field(default_factory=lambda: int(_env("MAX_CHARS", "2000")))
    max_revisions: int = field(default_factory=lambda: int(_env("MAX_REVISIONS", "2")))

    # Cadence: the scheduler fires daily; a run only proceeds if the last digest is older than this.
    min_hours_between: float = field(default_factory=lambda: float(_env("MIN_HOURS_BETWEEN", "47")))
    run_hour: int = field(default_factory=lambda: int(_env("RUN_HOUR", "8")))
    timezone: str = field(default_factory=lambda: _env("TZ", "UTC"))
    dedupe_days: int = field(default_factory=lambda: int(_env("DEDUPE_DAYS", "14")))

    # Signal (bbernhard/signal-cli-rest-api).
    signal_api_url: str = field(default_factory=lambda: _env("SIGNAL_API_URL", "http://localhost:8080"))
    signal_number: str | None = field(default_factory=lambda: _env("SIGNAL_NUMBER"))
    signal_group_id: str | None = field(default_factory=lambda: _env("SIGNAL_GROUP_ID"))

    # Ops.
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    healthcheck_url: str | None = field(default_factory=lambda: _env("HEALTHCHECK_URL"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "digest.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


settings = Settings()
