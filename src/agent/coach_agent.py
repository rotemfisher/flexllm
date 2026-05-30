import sqlite3
from contextlib import contextmanager

from src.config import config
from src.agent.graph import build_multi_agent_graph


def get_athlete_context() -> str:
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        profile = con.execute(
            "SELECT fitness_level, current_goal, secondary_goal FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
    except Exception as exc:
        return f"(Unable to load athlete profile: {exc})"

    if not profile:
        return "(No athlete profile found — call get_onboarding_status to set up.)"

    lines = ["=== ATHLETE PROFILE ==="]
    lines.append(f"Fitness level: {profile['fitness_level']}")
    if profile["current_goal"]:
        lines.append(f"Primary goal: {profile['current_goal']}")
    if profile["secondary_goal"]:
        lines.append(f"Secondary goal: {profile['secondary_goal']}")
    return "\n".join(lines)


@contextmanager
def build_coach_graph():
    with build_multi_agent_graph() as graph:
        yield graph
