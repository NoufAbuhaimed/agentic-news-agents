"""Small SQLite store: which links we've already sent, and the run history used for the cadence gate."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

_TRACKING_PARAMS = {"ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sent_items (
    url      TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    sent_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_stats (
    domain     TEXT PRIMARY KEY,         -- long-term memory: which sources produce published news
    published  INTEGER NOT NULL,
    last_used  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    status      TEXT NOT NULL,          -- running | sent | failed
    finished_at TEXT
);
"""


# Two links to the same page often differ slightly (tracking params, trailing slash, www).
# Normalizing them lets memory and dedupe recognize the same page.
def normalize_url(url: str) -> str:
    """Canonical form for dedupe: lowercase host, no fragment, no tracking params, no trailing slash."""
    parts = urlsplit(url.strip())
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query) if k not in _TRACKING_PARAMS and not k.startswith("utm_")]
    )
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- Long-term memory (SQLite file data/digest.sqlite) -------------------------------------------
# sent_items  : links already posted (never post twice; used by check_already_sent and select)
# source_stats: which websites produced posted news (the planner favors them)
# runs        : history of runs with status and cost (cadence, crash recovery, observability)
class Store:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(runs)")}
        if "cost_usd" not in cols:
            self.conn.execute("ALTER TABLE runs ADD COLUMN cost_usd REAL")

    # --- sent items -------------------------------------------------------
    def recent_urls(self, days: int) -> list[str]:
        cutoff = (_now() - timedelta(days=days)).isoformat()
        rows = self.conn.execute("SELECT url FROM sent_items WHERE sent_at >= ?", (cutoff,))
        return [r[0] for r in rows]

    def record_sent(self, items: list[tuple[str, str]]) -> None:
        now = _now().isoformat()
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO sent_items (url, title, sent_at) VALUES (?, ?, ?)",
                [(normalize_url(url), title, now) for url, title in items],
            )
            for url, _ in items:
                domain = urlsplit(url).netloc.lower().removeprefix("www.")
                self.conn.execute(
                    "INSERT INTO source_stats (domain, published, last_used) VALUES (?, 1, ?) "
                    "ON CONFLICT(domain) DO UPDATE SET published = published + 1, last_used = excluded.last_used",
                    (domain, now),
                )

    def top_sources(self, limit: int = 10) -> list[str]:
        rows = self.conn.execute(
            "SELECT domain FROM source_stats ORDER BY published DESC, last_used DESC LIMIT ?", (limit,)
        )
        return [r[0] for r in rows]

    # --- runs -------------------------------------------------------------
    def last_sent_at(self) -> datetime | None:
        row = self.conn.execute(
            "SELECT finished_at FROM runs WHERE status = 'sent' ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def unfinished_run(self, within_hours: float = 20) -> str | None:
        """A recent run that crashed or failed mid-graph, which we resume instead of restarting."""
        cutoff = (_now() - timedelta(hours=within_hours)).isoformat()
        row = self.conn.execute(
            "SELECT run_id FROM runs WHERE status IN ('running', 'failed') AND started_at >= ? "
            "ORDER BY started_at DESC LIMIT 1",
            (cutoff,),
        ).fetchone()
        return row[0] if row else None

    def start_run(self, run_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, started_at, status) VALUES (?, ?, 'running')",
                (run_id, _now().isoformat()),
            )
            self.conn.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))

    def finish_run(self, run_id: str, status: str, cost_usd: float | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, cost_usd = COALESCE(?, cost_usd) WHERE run_id = ?",
                (status, _now().isoformat(), cost_usd, run_id),
            )
