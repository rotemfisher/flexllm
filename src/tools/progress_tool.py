import sqlite3
from langchain_core.tools import tool
from src.config import config


def _fmt_pace(decimal_min: float | None) -> str:
    if decimal_min is None:
        return "N/A"
    minutes = int(decimal_min)
    seconds = round((decimal_min - minutes) * 60)
    return f"{minutes}:{seconds:02d}/km"


@tool
def get_progress_report(weeks: int = 8) -> str:
    """
    Generate a structured progress analysis covering the last N weeks.
    Use this for progress reviews, trend insights, or when the athlete asks
    how their training is going over time. Also useful before building a new plan.

    Args:
        weeks: number of weeks to look back (1–52, default 8)
    """
    weeks = min(max(weeks, 1), 52)

    con = None
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        weekly = con.execute(
            """
            SELECT
                strftime('%Y-W%W', start_date)           AS week,
                MIN(substr(start_date, 1, 10))           AS week_from,
                COUNT(*)                                 AS runs,
                ROUND(SUM(distance_km), 1)               AS total_km,
                ROUND(AVG(pace_min_per_km), 2)           AS avg_pace,
                ROUND(AVG(avg_heart_rate_bpm), 0)        AS avg_hr,
                ROUND(AVG(CAST(rpe AS REAL)), 1)         AS avg_rpe,
                ROUND(SUM(training_stress_score), 0)     AS total_tss
            FROM v_running_overview
            WHERE start_date >= date('now', ?)
            GROUP BY week
            ORDER BY week DESC
            """,
            (f"-{weeks} weeks",),
        ).fetchall()

        fitness = con.execute(
            """
            SELECT date, ctl, atl, tsb, hrv_sdnn_ms, vo2max_ml_kg_min,
                   resting_heart_rate_bpm, body_mass_kg
            FROM daily_health
            WHERE ctl IS NOT NULL
            ORDER BY date DESC LIMIT 1
            """
        ).fetchone()

        injuries = con.execute(
            """
            SELECT
                COUNT(*)                                                        AS total,
                SUM(CASE WHEN status IN ('active','recovering') THEN 1 ELSE 0 END) AS active_count
            FROM injuries
            WHERE onset_date >= date('now', ?)
            """,
            (f"-{weeks} weeks",),
        ).fetchone()

        lines = [f"=== Progress Report: Last {weeks} Weeks ===\n"]

        # --- Fitness snapshot ---
        if fitness:
            tsb = fitness["tsb"] or 0.0
            form = (
                "very fresh"        if tsb > 25  else
                "race-ready"        if tsb > 5   else
                "normal training"   if tsb > -10 else
                "building / fatigued" if tsb > -25 else
                "HIGH FATIGUE — reduce load"
            )
            lines.append("Current Fitness Snapshot:")
            lines.append(f"  CTL (fitness base) : {fitness['ctl']:.1f}")
            lines.append(f"  ATL (fatigue)      : {fitness['atl']:.1f}")
            lines.append(f"  TSB (form)         : {tsb:.1f}  →  {form}")
            if fitness["hrv_sdnn_ms"]:
                lines.append(f"  HRV                : {fitness['hrv_sdnn_ms']:.0f} ms")
            if fitness["vo2max_ml_kg_min"]:
                lines.append(f"  VO2max (estimated) : {fitness['vo2max_ml_kg_min']:.1f} ml/kg/min")
            if fitness["resting_heart_rate_bpm"]:
                lines.append(f"  Resting HR         : {fitness['resting_heart_rate_bpm']:.0f} bpm")
            lines.append("")

        # --- Weekly breakdown ---
        if weekly:
            lines.append("Weekly Running Breakdown (newest first):")
            header = f"  {'Date':<12} {'Runs':>4} {'km':>6} {'Pace':>9} {'HR':>5} {'RPE':>4} {'TSS':>5}"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for w in weekly:
                pace_s = _fmt_pace(w["avg_pace"])
                hr_s   = f"{w['avg_hr']:.0f}"   if w["avg_hr"]   else "N/A"
                rpe_s  = f"{w['avg_rpe']:.1f}"  if w["avg_rpe"]  else "N/A"
                tss_s  = f"{w['total_tss']:.0f}" if w["total_tss"] else "N/A"
                lines.append(
                    f"  {w['week_from']:<12} {w['runs']:>4} {w['total_km']:>6.1f} "
                    f"{pace_s:>9} {hr_s:>5} {rpe_s:>4} {tss_s:>5}"
                )
            lines.append("")

            # --- Trend signals ---
            if len(weekly) >= 2:
                latest_km = weekly[0]["total_km"] or 0.0
                prev_km   = weekly[1]["total_km"] or 0.0
                vol_delta = latest_km - prev_km
                vol_sign  = "+" if vol_delta >= 0 else ""
                lines.append(f"Volume trend  : {vol_sign}{vol_delta:.1f} km this week vs last week")

            oldest = next((w for w in reversed(weekly) if w["avg_pace"]), None)
            newest = next((w for w in weekly          if w["avg_pace"]), None)
            if oldest and newest and oldest["week"] != newest["week"]:
                pace_delta = oldest["avg_pace"] - newest["avg_pace"]
                if abs(pace_delta) >= 0.05:
                    direction = "faster" if pace_delta > 0 else "slower"
                    lines.append(
                        f"Pace trend    : {_fmt_pace(abs(pace_delta))} {direction} "
                        f"than {weeks} weeks ago"
                    )

            oldest_hr = next((w for w in reversed(weekly) if w["avg_hr"]), None)
            newest_hr = next((w for w in weekly          if w["avg_hr"]), None)
            if oldest_hr and newest_hr and oldest_hr["week"] != newest_hr["week"]:
                hr_delta = newest_hr["avg_hr"] - oldest_hr["avg_hr"]
                if abs(hr_delta) >= 2:
                    direction = "higher" if hr_delta > 0 else "lower"
                    lines.append(
                        f"HR trend      : avg HR {abs(hr_delta):.0f} bpm {direction} "
                        f"than {weeks} weeks ago (same pace → aerobic efficiency signal)"
                    )
            lines.append("")

        # --- Injury summary ---
        if injuries and injuries["total"] > 0:
            lines.append(
                f"Injuries (last {weeks} wks): "
                f"{injuries['total']} recorded, {injuries['active_count']} currently active/recovering"
            )
        else:
            lines.append(f"Injuries (last {weeks} wks): none recorded — training continuity maintained")

        return "\n".join(lines)

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()
