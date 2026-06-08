"""
Conversation memory: daily and weekly summary storage + generation.

Architecture
------------
After each CLI session (or explicitly triggered), coaching messages are
compressed into concise daily summaries stored in the `conversation_summaries`
table.  A weekly rollup is generated from those daily notes.

At session start, SummaryStore.format_for_context() injects the most recent
summaries into athlete_context so every agent has long-term memory without
requiring the full raw message history.

Summary hierarchy
-----------------
  daily   — one row per (date, domain): bullet-point session notes
  weekly  — one row per (week_start, "all"): consolidated weekly view
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row
from datetime import date, timedelta
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama


# ── Helpers ───────────────────────────────────────────────────────────────────

def _week_start(d: date) -> date:
    """Return the Sunday that begins the week containing *d* (Israeli convention)."""
    return d - timedelta(days=(d.weekday() + 1) % 7)


def _messages_to_text(messages: list[BaseMessage]) -> str:
    """Convert a message list to a plain-text transcript for summarisation."""
    lines = []
    for m in messages:
        content = getattr(m, "content", None)
        if not content:
            continue
        role = m.type.upper()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── SummaryStore ──────────────────────────────────────────────────────────────

class SummaryStore:
    """PostgreSQL-backed store for daily and weekly coaching summaries."""

    def __init__(self) -> None:
        from src.config import config
        self._dsn = config.DATABASE_URL
        self._init_schema()

    def _init_schema(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id           BIGSERIAL PRIMARY KEY,
                    summary_date TEXT      NOT NULL,
                    week_start   TEXT      NOT NULL,
                    domain       TEXT      NOT NULL,
                    summary_type TEXT      NOT NULL,
                    content      TEXT      NOT NULL,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(summary_date, domain, summary_type)
                )
            """)

    def save(
        self,
        summary_date: date,
        domain: str,
        summary_type: str,
        content: str,
    ) -> None:
        ws = _week_start(summary_date)
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO conversation_summaries
                    (summary_date, week_start, domain, summary_type, content)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (summary_date, domain, summary_type) DO UPDATE
                    SET content    = EXCLUDED.content,
                        week_start = EXCLUDED.week_start
                """,
                (summary_date.isoformat(), ws.isoformat(), domain, summary_type, content),
            )

    def get_recent_daily(self, days: int = 7) -> list[dict]:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT summary_date, domain, content
                FROM   conversation_summaries
                WHERE  summary_type = 'daily' AND summary_date >= %s
                ORDER  BY summary_date DESC, domain
                """,
                (cutoff,),
            ).fetchall()
        return [{"date": r["summary_date"], "domain": r["domain"], "content": r["content"]} for r in rows]

    def get_current_week_summary(self) -> Optional[str]:
        ws = _week_start(date.today()).isoformat()
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT content FROM conversation_summaries
                WHERE  summary_type = 'weekly' AND week_start = %s
                ORDER  BY created_at DESC LIMIT 1
                """,
                (ws,),
            ).fetchone()
        return row["content"] if row else None

    def format_for_context(self) -> str:
        """Return a compact context block to prepend to athlete_context.

        Includes the current weekly summary (if present) followed by
        daily session notes from the last 7 days.
        """
        parts: list[str] = []

        weekly = self.get_current_week_summary()
        if weekly:
            parts.append("=== THIS WEEK'S COACHING SUMMARY ===")
            parts.append(weekly)

        daily = self.get_recent_daily(days=7)
        if daily:
            parts.append("=== RECENT SESSION NOTES (last 7 days) ===")
            for d in daily:
                parts.append(f"[{d['date']} / {d['domain']}] {d['content']}")

        return "\n".join(parts)


# ── Summary generation ────────────────────────────────────────────────────────

def generate_daily_summary(
    messages: list[BaseMessage],
    domain: str,
    model_id: str,
) -> str:
    """Compress a session's messages into 3–5 bullet-point daily notes.

    Returns an empty string if there is nothing meaningful to summarise.
    """
    transcript = _messages_to_text(messages)
    if not transcript.strip():
        return ""

    llm = ChatOllama(model=model_id, temperature=0)
    prompt = (
        f"Summarise the following {domain} coaching session in 3–5 concise bullet points.\n"
        "Focus on: key findings, decisions made, recommendations given, action items.\n"
        "Do not include greetings, meta-commentary, or repeated information.\n\n"
        f"--- SESSION ---\n{transcript}\n--- END ---"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def generate_weekly_summary(
    daily_summaries: list[dict],
    model_id: str,
) -> str:
    """Roll up a week's daily summaries into a consolidated weekly overview.

    Returns an empty string when called with no data.
    """
    if not daily_summaries:
        return ""

    notes = "\n".join(
        f"[{d['date']} / {d['domain']}] {d['content']}"
        for d in daily_summaries
    )
    llm = ChatOllama(model=model_id, temperature=0)
    prompt = (
        "Create a structured weekly coaching summary from the daily notes below.\n"
        "Cover (omit any category with no data):\n"
        "  • Training volume and progression\n"
        "  • Performance trends\n"
        "  • Recovery status\n"
        "  • Injury considerations\n"
        "  • Nutrition adherence\n"
        "  • Key coaching decisions\n"
        "3–6 bullet points per category, factual and concise.\n\n"
        f"--- DAILY NOTES ---\n{notes}\n--- END ---"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ── Convenience: end-of-session persistence ───────────────────────────────────

def save_session_summary(
    messages: list[BaseMessage],
    domain: str,
    store: SummaryStore,
    model_id: str,
    session_date: Optional[date] = None,
) -> None:
    """Generate and persist a daily summary for one agent domain.

    A minimum of 4 messages is required to justify summarisation (system +
    human + at least one AI reply + one more exchange).
    """
    if len(messages) < 4:
        return
    if session_date is None:
        session_date = date.today()
    summary = generate_daily_summary(messages, domain, model_id)
    if summary:
        store.save(session_date, domain, "daily", summary)


def maybe_refresh_weekly_summary(store: SummaryStore, model_id: str) -> None:
    """Regenerate the weekly summary if we have daily data but no weekly yet."""
    if store.get_current_week_summary():
        return
    daily = store.get_recent_daily(days=7)
    if len(daily) < 2:
        return
    weekly = generate_weekly_summary(daily, model_id)
    if weekly:
        store.save(date.today(), "all", "weekly", weekly)
