import sqlite3
import json
from langchain_core.tools import tool
from src.config import config

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
    con = None
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        rows = [dict(r) for r in con.execute(
            """
            SELECT start_date, duration_min, distance_km,
                   ROUND(duration_min / NULLIF(distance_km, 0), 2) AS pace_min_per_km,
                   avg_heart_rate_bpm, training_stress_score, rpe
            FROM workouts
            WHERE activity_type = ?
            ORDER BY start_date DESC LIMIT ?
            """,
            (activity_type, limit)
        ).fetchall()]

        return json.dumps(rows, default=str) if rows else f"No recent {activity_type} workouts found."

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()
