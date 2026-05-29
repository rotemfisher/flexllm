import sqlite3
from datetime import datetime
from langchain_core.tools import tool
from src.config import config

@tool
def get_daily_readiness(date: str | None = None) -> str:
    """
    Get the athlete's recovery, sleep, and training load status for a specific date.
    Use this to assess fatigue, recovery, and readiness before prescribing a workout.

    Args:
        date (str, optional): The date in 'YYYY-MM-DD' format. Defaults to today.
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    con = None
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        health = con.execute(
            "SELECT * FROM daily_health WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (date,)
        ).fetchone()

        if not health:
            return f"No health or readiness data found on or before {date}."

        sleep_total = health['sleep_total_min'] or 0
        sleep_hours = sleep_total // 60
        sleep_mins = sleep_total % 60

        atl = health['atl'] or 0.0
        ctl = health['ctl'] or 0.0
        tsb = health['tsb'] or 0.0

        form_status = "Fresh" if tsb > 0 else "Fatigued" if tsb < -10 else "Optimal/Neutral"

        return (
            f"--- Readiness Report for {health['date']} ---\n"
            f"Training Load (Fitness & Fatigue):\n"
            f"  - CTL (Fitness): {ctl:.1f}\n"
            f"  - ATL (Fatigue): {atl:.1f}\n"
            f"  - TSB (Form):    {tsb:.1f} ({form_status})\n\n"
            f"Physiological Markers:\n"
            f"  - Resting HR:    {health['resting_heart_rate_bpm'] or 'N/A'} bpm\n"
            f"  - HRV (SDNN):    {health['hrv_sdnn_ms'] or 'N/A'} ms\n"
            f"  - Body Mass:     {health['body_mass_kg'] or 'N/A'} kg\n\n"
            f"Sleep (Previous Night):\n"
            f"  - Total Sleep:   {sleep_hours}h {sleep_mins}m\n"
            f"  - Deep Sleep:    {health['sleep_deep_min'] or 0} min\n"
            f"  - REM Sleep:     {health['sleep_rem_min'] or 0} min\n"
        )

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()
