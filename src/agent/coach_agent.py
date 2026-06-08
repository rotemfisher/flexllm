import sqlite3
from contextlib import asynccontextmanager

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

    # ── Recent workouts (last 8 weeks) ───────────────────────────────────────
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        workouts = con.execute(
            """
            SELECT activity_type, start_date, distance_km, duration_min,
                   avg_heart_rate_bpm, avg_speed_kmh, training_stress_score
            FROM workouts
            WHERE start_date >= date('now', '-56 days')
            ORDER BY start_date DESC
            LIMIT 20
            """
        ).fetchall()
        con.close()

        if workouts:
            lines = ["=== RECENT WORKOUTS (last 8 weeks) ==="]
            for w in workouts:
                date = str(w["start_date"])[:10]
                dist = f"{w['distance_km']:.2f} km" if w["distance_km"] else "—"
                dur  = f"{w['duration_min']:.0f} min" if w["duration_min"] else "—"
                hr   = f"{w['avg_heart_rate_bpm']:.0f} bpm" if w["avg_heart_rate_bpm"] else "—"
                pace = (
                    f"{60 / w['avg_speed_kmh']:.2f} min/km"
                    if w["avg_speed_kmh"] else "—"
                )
                tss  = f"TSS {w['training_stress_score']:.0f}" if w["training_stress_score"] else ""
                lines.append(
                    f"  {date}  {w['activity_type']:12s}  {dist:>9}  {dur:>7}  "
                    f"pace {pace}  HR {hr}  {tss}"
                )
            parts.append("\n".join(lines))
    except Exception as exc:
        logger.error("Failed to load recent workouts", exc_info=True)

    # ── Conversation summaries (long-term memory) ─────────────────────────────
    try:
        store = SummaryStore(str(config.DB_PATH))
        summary_block = store.format_for_context()
        if summary_block:
            parts.append(summary_block)
    except Exception as exc:
        logger.error("Failed to load summaries", exc_info=True)
        pass  # Summaries are best-effort; never block session startup.

    return "\n\n".join(parts)


@asynccontextmanager
async def build_coach_graph():
    async with build_multi_agent_graph() as graph:
        yield graph
