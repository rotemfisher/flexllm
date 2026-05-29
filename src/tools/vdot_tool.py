import sqlite3
from langchain_core.tools import tool
from src.config import config


def _fmt(seconds: int | None) -> str:
    if seconds is None:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}/km"


@tool
def get_vdot_paces(vdot: int) -> str:
    """
    Look up Daniels' training paces for a given VDOT score (30–85).
    Returns Easy, Marathon, Threshold, Interval, and Repetition paces in min:sec/km.

    Use this whenever the athlete asks about target training paces or what pace
    they should run for a specific workout type given their current VDOT.
    """
    if not (30 <= vdot <= 85):
        return f"VDOT must be between 30 and 85. Received: {vdot}"

    con = None
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM vdot_paces WHERE vdot = ?", (vdot,)
        ).fetchone()

        if not row:
            return f"No paces found for VDOT {vdot}."

        return (
            f"VDOT {vdot} — Daniels Training Paces:\n"
            f"  Easy:        {_fmt(row['e_pace_slow_sec'])} – {_fmt(row['e_pace_fast_sec'])}\n"
            f"  Marathon:    {_fmt(row['m_pace_sec'])}\n"
            f"  Threshold:   {_fmt(row['t_pace_sec'])}\n"
            f"  Interval:    {_fmt(row['i_pace_sec'])}\n"
            f"  Repetition:  {_fmt(row['r_pace_sec'])}"
        )

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()
