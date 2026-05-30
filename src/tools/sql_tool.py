import json
import re

from langchain_core.tools import tool

from src.tools._utils import db_ro

_WRITE_OPS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)


@tool
def query_running_database(query: str) -> str:
    """
    Execute a read-only SQL query against the athlete's training database.

    Key tables:
    - v_running_overview  Running-only view — use first for running questions.
      Columns: id, start_date, duration_min, distance_km, pace_min_per_km,
               avg_heart_rate_bpm, max_heart_rate_bpm, training_stress_score,
               atl, ctl, tsb, ground_contact_avg_ms, running_power_avg_w,
               resting_heart_rate_bpm, hrv_sdnn_ms, sleep_total_min, shoe
    - workouts            All sessions (running, strength, cycling, walking).
      Key columns: activity_type, start_date, distance_km, duration_min,
                   avg_heart_rate_bpm, training_stress_score
    - daily_health        Per-day ATL, CTL, TSB, HRV, resting_hr, sleep totals
    - sleep_records       Nightly stages: deep, REM, core, awake (duration_min each)
    - kilometer_splits    Per-km pace and HR for individual runs
    - vdot_paces          Daniels training paces by VDOT score

    Rules:
    - Add LIMIT (enforced at max 50 if missing)
    - Dates are TEXT: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
    - For most running questions start with v_running_overview
    """
    if _WRITE_OPS.search(query):
        return "Error: only SELECT queries are permitted."

    if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        query = query.rstrip("; ") + " LIMIT 50"

    try:
        with db_ro() as con:
            rows = [dict(r) for r in con.execute(query).fetchall()]
        return json.dumps(rows, default=str) if rows else "No results found."
    except Exception as exc:
        return f"Query error: {exc}"
