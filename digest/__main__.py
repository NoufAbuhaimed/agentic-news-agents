"""CLI.

  python -m digest run [--dry-run] [--force]   one run (respects the every-other-day gate)
  python -m digest schedule                    long-running: fires `run` daily at RUN_HOUR
  python -m digest graph                       show the architecture (graph + agent) from the code
  python -m digest guardrails                  offline demo of the guard rails (free)
  python -m digest groups                      list Signal groups (to find SIGNAL_GROUP_ID)
  python -m digest test-send "hello"           send a test message to the group
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import costs
from .config import settings
from .observability import StepLogger, ping, setup_logging

log = logging.getLogger("digest")


def run(dry_run: bool = False, force: bool = False, on_update=None, callbacks=(), since: str | None = None) -> dict | None:
    """Run the digest graph once. Returns {run_id, result, cost} or None when not due yet.

    on_update(node, update) is called after every graph step (used by the notebook to show progress).
    """
    from .graph import build_graph
    from .nodes import store

    since_override = since
    db = store()
    now = datetime.now(timezone.utc)
    last = db.last_sent_at()
    if not (dry_run or force) and last and now - last < timedelta(hours=settings.min_hours_between):
        log.info("last digest sent %s; not due yet", last.isoformat(timespec="minutes"))
        return None

    graph = build_graph()
    resume_id = None if dry_run else db.unfinished_run()
    run_id = resume_id or f"{'dry-' if dry_run else ''}{now:%Y%m%d-%H%M}-{uuid.uuid4().hex[:6]}"
    tracker = costs.start(settings.run_budget_usd)
    config = {
        "configurable": {"thread_id": run_id},
        "callbacks": [StepLogger(), tracker, *callbacks],
        # LangSmith picks these up when LANGSMITH_TRACING=true.
        "run_name": f"digest {now:%Y-%m-%d}",
        "tags": ["tech-digest", "dry-run" if dry_run else "live"],
        "metadata": {"run_id": run_id, "resumed": bool(resume_id)},
        "max_concurrency": 6,
    }

    if not dry_run:
        db.start_run(run_id)
        ping("/start")
    def execute(graph_input):
        if on_update is None:
            return graph.invoke(graph_input, config)
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                on_update(node, update)
        return graph.get_state(config).values

    try:
        if resume_id:
            log.info("resuming unfinished run %s from its last checkpoint", run_id)
            result = execute(None)
        else:
            today = datetime.now(ZoneInfo(settings.timezone)).date()
            # One day of overlap so nothing falls between runs; dedupe stops repeats.
            since = (last.date() if last else today - timedelta(days=2)) - timedelta(days=1)
            if since_override:
                since = date.fromisoformat(since_override)
            log.info("starting run %s (news since %s)", run_id, since)
            result = execute(
                {
                    "run_id": run_id,
                    "today": today.isoformat(),
                    "since": since.isoformat(),
                    "recent_urls": db.recent_urls(settings.dedupe_days),
                    "dry_run": dry_run,
                    "candidates": [],
                    "checked": [],
                    "fact_checked": [],
                    "attempts": {},
                    "fact_notes": {},
                    "revisions": 0,
                }
            )
    except Exception:
        log.exception("run %s failed; next run will resume from the last checkpoint", run_id)
        log.info("run cost so far: %s", tracker.summary())
        if not dry_run:
            db.finish_run(run_id, "failed", tracker.total_usd)
            ping("/fail")
        raise

    log.info("run cost: %s (budget $%.2f)", tracker.summary(), settings.run_budget_usd)
    if not dry_run:
        # Only a posted digest moves the date window and the every-other-day timer; an empty run
        # ("nothing new") is recorded separately so tomorrow's check tries again.
        db.finish_run(run_id, "sent" if result.get("sent") else "empty", tracker.total_usd)
        ping()
    return {"run_id": run_id, "result": result, "cost": tracker}


def schedule() -> None:
    tz = ZoneInfo(settings.timezone)
    log.info("scheduler up: daily at %02d:00 %s (min %sh between digests)",
             settings.run_hour, settings.timezone, settings.min_hours_between)
    while True:
        now = datetime.now(tz)
        next_run = now.replace(hour=settings.run_hour, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        log.info("next check at %s", next_run.isoformat(timespec="minutes"))
        time.sleep((next_run - now).total_seconds())
        try:
            run()
        except Exception:
            pass  # already logged and reported; keep the scheduler alive


def main() -> None:
    p = argparse.ArgumentParser(prog="digest")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true", help="research and write, print instead of sending")
    r.add_argument("--force", action="store_true", help="ignore the every-other-day gate")
    r.add_argument("--since", help="override the start of the news window (YYYY-MM-DD), e.g. for demos")
    sub.add_parser("schedule")
    sub.add_parser("graph", help="show the architecture, built from the code")
    sub.add_parser("guardrails", help="offline demo of the guard rails")
    sub.add_parser("groups")
    t = sub.add_parser("test-send")
    t.add_argument("message")
    args = p.parse_args()

    if args.cmd in ("graph", "guardrails"):
        from . import demo
        demo.show_graph() if args.cmd == "graph" else demo.show_guardrails()
        return

    setup_logging()
    if args.cmd == "run":
        from .demo import print_summary
        from .observability import RootRunCapture

        trace = RootRunCapture()
        out = run(dry_run=args.dry_run, force=args.force, callbacks=[trace], since=args.since)
        if out:
            print_summary(out, trace.run_id)
    elif args.cmd == "schedule":
        schedule()
    elif args.cmd == "groups":
        from .signal_client import list_groups
        for grp in list_groups():
            print(f"{grp.get('id')}\t{grp.get('name')}")
    elif args.cmd == "test-send":
        from .signal_client import send
        send(args.message)
        print("sent")


if __name__ == "__main__":
    main()
