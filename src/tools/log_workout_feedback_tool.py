import sqlite3
from langchain_core.tools import tool
from src.config import config

@tool
def log_workout_rpe_and_notes(workout_id: int, rpe: int, notes: str) -> str:
    """
    Use this tool to save the athlete's subjective feedback (RPE and notes)
    after they tell you how a recent workout felt.

    Args:
        workout_id (int): The ID of the workout (fetch this from get_recent_workouts first).
        rpe (int): Rate of Perceived Exertion (1 to 10).
        notes (str): The athlete's subjective feedback.
    """
    if not (1 <= rpe <= 10):
        return "Error: RPE must be an integer between 1 and 10."

    con = None
    try:
        con = sqlite3.connect(config.DB_PATH)

        exists = con.execute("SELECT id FROM workouts WHERE id = ?", (workout_id,)).fetchone()
        if not exists:
            return f"Error: Workout ID {workout_id} does not exist."

        con.execute(
            "UPDATE workouts SET rpe = ?, notes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (rpe, notes, workout_id)
        )
        con.commit()
        return f"Successfully logged RPE {rpe} and notes for workout {workout_id}."

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()
