import sqlite3
from contextlib import contextmanager

from src.config import config
from src.agent.graph import build_multi_agent_graph
from src.agent.memory import SummaryStore
from src.tracing import traceable
import logging

logger = logging.getLogger(__name__)


@traceable(name="get_athlete_context", run_type="retriever")
def get_athlete_context() -> str:
    """Build the static context block injected into every agent's system prompt.

    Includes the basic athlete profile plus any persisted summaries so agents
    have long-term memory across sessions without loading the full message history.
    """
    parts: list[str] = []

    # ── Athlete profile ───────────────────────────────────────────────────────
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        profile = con.execute(
            "SELECT fitness_level, current_goal, secondary_goal FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
    except Exception as exc:
        logger.error("Failed to load profile", exc_info=True)
        return f"(Unable to load athlete profile: {exc})"

    if not profile:
        return "(No athlete profile found — call get_onboarding_status to set up.)"

    lines = ["=== ATHLETE PROFILE ==="]
    lines.append(f"Fitness level: {profile['fitness_level']}")
    if profile["current_goal"]:
        lines.append(f"Primary goal:  {profile['current_goal']}")
    if profile["secondary_goal"]:
        lines.append(f"Secondary goal: {profile['secondary_goal']}")
    parts.append("\n".join(lines))

    # ── Conversation summaries (long-term memory) ─────────────────────────────
    try:
        store = SummaryStore(config.DB_PATH)
        summary_block = store.format_for_context()
        if summary_block:
            parts.append(summary_block)
    except Exception as exc:
        logger.error("Failed to load summaries", exc_info=True)
        pass  # Summaries are best-effort; never block session startup.

    return "\n\n".join(parts)


@contextmanager
def build_coach_graph():
    with build_multi_agent_graph() as graph:
        yield graph
