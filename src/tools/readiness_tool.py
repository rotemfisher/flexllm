import logging
from datetime import date, datetime, timezone

from langchain_core.tools import tool

from src.tools._utils import db_ro

logger = logging.getLogger(__name__)

# If the latest daily_health row is older than this many days, the agent must
# ask the user to sync their Apple Watch before giving any readiness advice.
_STALE_DAYS = 7


def _stale_warning(data_date: str) -> str:
    """Return a non-empty warning string if data_date is older than _STALE_DAYS."""
    try:
        age = (datetime.now(timezone.utc).date() - date.fromisoformat(data_date)).days
    except (ValueError, TypeError):
        return ""
    if age <= _STALE_DAYS:
        return ""
    return (
        f"⚠️  DATA IS STALE — The most recent health record is {age} day(s) old "
        f"(last sync: {data_date}). Please ask the user to open the Health app on their iPhone, "
        f"ensure their Apple Watch has synced, then re-export and re-ingest their data before "
        f"giving readiness or training-load advice based on these numbers.\n\n"
    )


@tool
def get_daily_readiness(date: str | None = None) -> str:
    """
    Get the athlete's recovery, sleep, and training load status for a specific date.
    Use this to assess fatigue, recovery, and readiness before prescribing a workout.

    Args:
        date (str, optional): The date in 'YYYY-MM-DD' format. Defaults to today.
    """
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        with db_ro() as con:
            health = con.execute(
                """
                SELECT date, atl, ctl, tsb,
                       resting_heart_rate_bpm, hrv_sdnn_ms, body_mass_kg,
                       sleep_total_min, sleep_deep_min, sleep_rem_min
                FROM daily_health
                WHERE date <= %s
                ORDER BY date DESC LIMIT 1
                """,
                (date,),
            ).fetchone()

        if not health:
            return f"No health or readiness data found on or before {date}."

        warning = _stale_warning(health["date"])

        sleep_total = health["sleep_total_min"] or 0
        sleep_hours = sleep_total // 60
        sleep_mins  = sleep_total % 60

        atl = health["atl"] or 0.0
        ctl = health["ctl"] or 0.0
        tsb = health["tsb"] or 0.0

        form_status = "Fresh" if tsb > 0 else "Fatigued" if tsb < -10 else "Optimal/Neutral"

        return warning + (
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
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"
