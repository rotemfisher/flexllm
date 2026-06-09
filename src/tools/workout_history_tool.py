import logging

from langchain_core.tools import tool

from src.tools._utils import db_ro

logger = logging.getLogger(__name__)


def _fmt_pace(decimal_min: float | None) -> str:
    if decimal_min is None:
        return "N/A"
    minutes = int(decimal_min)
    seconds = round((decimal_min - minutes) * 60)
    return f"{minutes}:{seconds:02d}/km"


@tool
def get_recent_workouts(limit: int = 5, activity_type: str = "running") -> str:
    """
    ALWAYS use this tool first to check the athlete's recent training history.
    Returns the most recent workouts with their key metrics, pace, and RPE.

    Args:
        limit (int): Number of workouts to return (max 15). Defaults to 5.
        activity_type (str): Filter by type — 'running', 'strength', 'cycling', 'walking'. Defaults to 'running'.
    """
    limit = min(limit, 15)
    try:
        with db_ro() as con:
            rows = con.execute(
                """
                SELECT id, start_date, duration_min, distance_km,
                       ROUND((duration_min / NULLIF(distance_km, 0))::numeric, 2) AS pace_min_per_km,
                       avg_heart_rate_bpm, training_stress_score, rpe
                FROM workouts
                WHERE activity_type = %s
                ORDER BY start_date DESC LIMIT %s
                """,
                (activity_type, limit),
            ).fetchall()

        if not rows:
            return f"No recent {activity_type} workouts found."

        report = f"--- Recent {activity_type.capitalize()} Workouts ---\n"
        for r in rows:
            date_str = r["start_date"][:10]
            dist = f"{r['distance_km']}km" if r["distance_km"] else "N/A"
            dur  = f"{r['duration_min']}m" if r["duration_min"] else "N/A"
            pace = _fmt_pace(r["pace_min_per_km"])
            hr   = f"{round(r['avg_heart_rate_bpm'])} bpm" if r["avg_heart_rate_bpm"] else "N/A"
            rpe  = f"{r['rpe']}/10"                        if r["rpe"]                 else "Not logged"
            report += f"- {date_str} [ID: {r['id']}]: {dist} in {dur} | Pace: {pace} | HR: {hr} | RPE: {rpe}\n"

        return report

    except Exception as exc:
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"
