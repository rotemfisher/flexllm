import logging
from datetime import datetime, timezone

from langchain_core.tools import tool

from src.tools._utils import db_rw

logger = logging.getLogger(__name__)


@tool
def log_workout_rpe_and_notes(rpe: int, notes: str, date: str | None = None, activity_type: str = "running") -> str:
    """
    Use this tool to save the athlete's subjective feedback (RPE and notes) for a workout.

    Args:
        rpe (int): Rate of Perceived Exertion (1 to 10).
        notes (str): The athlete's subjective feedback.
        date (str, optional): The date of the workout in 'YYYY-MM-DD' format. If None, uses today's date (UTC).
        activity_type (str, optional): 'running', 'strength', etc. Defaults to 'running'.
    """
    if not (1 <= rpe <= 10):
        return "Error: RPE must be an integer between 1 and 10."

    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        with db_rw() as con:
            workout = con.execute(
                """
                SELECT id FROM workouts
                WHERE activity_type = ? AND start_date LIKE ?
                ORDER BY start_date DESC LIMIT 1
                """,
                (activity_type, f"{date}%"),
            ).fetchone()

            if not workout:
                return f"Error: No {activity_type} workout found on {date}. Cannot log RPE."

            con.execute(
                "UPDATE workouts SET rpe = ?, notes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (rpe, notes, workout[0]),
            )
            con.commit()
        return f"Successfully logged RPE {rpe} and notes for the {activity_type} workout on {date}."

    except Exception as exc:
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"
