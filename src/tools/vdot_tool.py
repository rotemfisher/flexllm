from langchain_core.tools import tool

from src.tools._utils import db_ro


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

    Note: not every integer VDOT has a row in the source table (e.g. 81–84 are
    absent from Daniels' Formula). When the exact value is missing the nearest
    available VDOT is returned with a note.
    """
    if not (30 <= vdot <= 85):
        return f"VDOT must be between 30 and 85. Received: {vdot}"

    try:
        with db_ro() as con:
            row = con.execute(
                "SELECT * FROM vdot_paces ORDER BY ABS(vdot - ?) ASC LIMIT 1", (vdot,)
            ).fetchone()

        if not row:
            return f"No paces found for VDOT {vdot}."

        actual = row["vdot"]
        note   = f" (nearest available in Daniels' table: VDOT {actual})" if actual != vdot else ""
        return (
            f"VDOT {vdot}{note} — Daniels Training Paces:\n"
            f"  Easy:        {_fmt(row['e_pace_slow_sec'])} – {_fmt(row['e_pace_fast_sec'])}\n"
            f"  Marathon:    {_fmt(row['m_pace_sec'])}\n"
            f"  Threshold:   {_fmt(row['t_pace_sec'])}\n"
            f"  Interval:    {_fmt(row['i_pace_sec'])}\n"
            f"  Repetition:  {_fmt(row['r_pace_sec'])}"
        )

    except Exception as exc:
        return f"Database error: {exc}"
