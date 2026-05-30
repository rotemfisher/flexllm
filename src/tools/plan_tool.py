import json
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from src.tools._utils import db_ro, db_rw


@tool
def save_workout_plan(week_start: str, sessions: str) -> str:
    """
    Save a weekly training plan to the database. Replaces any existing plan for that week.
    Call this after generating a workout plan for the athlete.

    Args:
        week_start: The Sunday that starts the plan week in 'YYYY-MM-DD' format.
        sessions: JSON array string where each object has:
            Required:
              - day_date (str, YYYY-MM-DD)
              - activity_type (str: 'running' | 'strength' | 'rest' | 'cross_training')
              - workout_type (str: 'easy' | 'tempo' | 'interval' | 'long_run' | 'recovery' |
                                   'strength' | 'rest' | 'assessment')
              - description (str: full session description)
              - intensity (str: 'easy' | 'moderate' | 'hard' | 'rest')
            Optional:
              - target_distance_km (float)
              - target_duration_min (float)
              - phase (str: 'onboarding'|'base'|'build'|'peak'|'race'|'recovery'|'return_to_run')
              - is_assessment (int: 1 for progress test / physical exam sessions, else 0)
              - notes (str: coach rationale)

    Example sessions value:
        '[{"day_date":"2026-06-02","activity_type":"running","workout_type":"easy",
           "description":"45 min easy run at E pace","intensity":"easy",
           "target_duration_min":45,"phase":"base"},
          {"day_date":"2026-06-05","activity_type":"running","workout_type":"assessment",
           "description":"2km time trial — record time for VDOT update",
           "intensity":"hard","is_assessment":1,"phase":"base"}]'
    """
    try:
        plan = json.loads(sessions)
    except json.JSONDecodeError as exc:
        return f"Error: sessions must be a valid JSON array. {exc}"

    required = {"day_date", "activity_type", "workout_type", "description", "intensity"}
    for i, s in enumerate(plan):
        missing = required - s.keys()
        if missing:
            return f"Error: session {i} is missing required fields: {sorted(missing)}"

    try:
        with db_rw() as con:
            # Explicit transaction: DELETE + INSERT must be atomic.
            con.execute("BEGIN")
            con.execute("DELETE FROM planned_workouts WHERE week_start = ?", (week_start,))
            for order, s in enumerate(plan, start=1):
                con.execute(
                    """
                    INSERT INTO planned_workouts
                        (week_start, day_date, session_order, activity_type, workout_type,
                         description, target_distance_km, target_duration_min,
                         intensity, phase, is_assessment, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        week_start,
                        s["day_date"],
                        s.get("session_order", order),
                        s["activity_type"],
                        s["workout_type"],
                        s["description"],
                        s.get("target_distance_km"),
                        s.get("target_duration_min"),
                        s["intensity"],
                        s.get("phase"),
                        int(s.get("is_assessment", 0)),
                        s.get("notes"),
                    ),
                )
            con.commit()
        assessment_count = sum(1 for s in plan if s.get("is_assessment"))
        summary = f"Saved {len(plan)} sessions for the week of {week_start}."
        if assessment_count:
            summary += f" ({assessment_count} assessment session(s) included.)"
        return summary
    except Exception as exc:
        return f"Database error: {exc}"


@tool
def get_current_workout_plan() -> str:
    """
    Retrieve the current week's training plan from the database.
    Call this at the start of each session to see what's scheduled,
    or when the athlete asks what they should do today or this week.
    Falls back to the next upcoming plan if nothing is saved for the current week.
    """
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    # Week starts on Sunday (Israeli convention). weekday(): Mon=0…Sat=5, Sun=6
    week_start = (now - timedelta(days=(now.weekday() + 1) % 7)).strftime("%Y-%m-%d")

    try:
        with db_ro() as con:
            rows = con.execute(
                """
                SELECT day_date, activity_type, workout_type, description,
                       target_distance_km, target_duration_min, intensity,
                       phase, is_assessment, notes, status
                FROM planned_workouts
                WHERE week_start = ?
                ORDER BY day_date, session_order
                """,
                (week_start,),
            ).fetchall()

            if not rows:
                rows = con.execute(
                    """
                    SELECT week_start, day_date, activity_type, workout_type, description,
                           target_distance_km, target_duration_min, intensity,
                           phase, is_assessment, notes, status
                    FROM planned_workouts
                    WHERE week_start > ?
                    ORDER BY week_start, day_date, session_order
                    LIMIT 7
                    """,
                    (week_start,),
                ).fetchall()
                if not rows:
                    return "No training plan found. Ask the coach to generate a weekly plan first."
                week_start = rows[0]["week_start"]

        phase_label = rows[0]["phase"] or "general"
        lines = [f"--- Training Plan: Week of {week_start} (Phase: {phase_label}) ---\n"]
        for row in rows:
            today_marker   = "  ← TODAY"            if row["day_date"] == today          else ""
            status_tag     = f" [{row['status'].upper()}]" if row["status"] != "planned" else ""
            assessment_tag = "  [ASSESSMENT/TEST]"  if row["is_assessment"]              else ""
            dist = f"  {row['target_distance_km']} km"       if row["target_distance_km"]  else ""
            dur  = f"  {row['target_duration_min']:.0f} min" if row["target_duration_min"] else ""
            lines.append(
                f"{row['day_date']}{today_marker}{status_tag}{assessment_tag}\n"
                f"  {row['workout_type'].title()} {row['activity_type']}{dist}{dur}\n"
                f"  {row['description']}"
                + (f"\n  Note: {row['notes']}" if row["notes"] else "")
            )
        return "\n\n".join(lines)

    except Exception as exc:
        return f"Database error: {exc}"


@tool
def replace_day_in_plan(week_start: str, day_date: str, sessions: str) -> str:
    """
    Replace all sessions for a single day within an existing weekly plan.
    Use this to adjust, swap, or add a session for one day without
    rewriting the entire week.

    Args:
        week_start: The Sunday that starts the plan week in 'YYYY-MM-DD' format.
        day_date: The specific day to update in 'YYYY-MM-DD' format.
        sessions: JSON array of session objects — same schema as save_workout_plan.
    """
    try:
        plan = json.loads(sessions)
    except json.JSONDecodeError as exc:
        return f"Error: sessions must be a valid JSON array. {exc}"

    required = {"activity_type", "workout_type", "description", "intensity"}
    for i, s in enumerate(plan):
        missing = required - s.keys()
        if missing:
            return f"Error: session {i} is missing required fields: {sorted(missing)}"

    try:
        with db_rw() as con:
            con.execute("BEGIN")
            con.execute(
                "DELETE FROM planned_workouts WHERE week_start = ? AND day_date = ?",
                (week_start, day_date),
            )
            for order, s in enumerate(plan, start=1):
                con.execute(
                    """
                    INSERT INTO planned_workouts
                        (week_start, day_date, session_order, activity_type, workout_type,
                         description, target_distance_km, target_duration_min,
                         intensity, phase, is_assessment, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        week_start,
                        day_date,
                        s.get("session_order", order),
                        s["activity_type"],
                        s["workout_type"],
                        s["description"],
                        s.get("target_distance_km"),
                        s.get("target_duration_min"),
                        s["intensity"],
                        s.get("phase"),
                        int(s.get("is_assessment", 0)),
                        s.get("notes"),
                    ),
                )
            con.commit()
        return f"Updated {len(plan)} session(s) for {day_date} in the week of {week_start}."
    except Exception as exc:
        return f"Database error: {exc}"


@tool
def update_planned_workout_status(
    day_date: str,
    status: str,
    reason: str | None = None,
    week_start: str | None = None,
) -> str:
    """
    Update the status of planned sessions on a specific date.
    Use this when sessions are skipped due to injury or fatigue, or marked as completed.

    Args:
        day_date: 'YYYY-MM-DD' the date of the session(s) to update.
        status: 'completed' | 'skipped' | 'modified'
        reason: why the status changed (e.g. 'knee injury flare', 'travel', 'coach modification')
        week_start: 'YYYY-MM-DD' of the week's Sunday — supply this to narrow the update
                    to a specific plan when the same date appears in multiple plans.
    """
    if status not in ("completed", "skipped", "modified"):
        return "Error: status must be 'completed', 'skipped', or 'modified'."

    try:
        with db_rw() as con:
            if week_start:
                count = con.execute(
                    "SELECT COUNT(*) FROM planned_workouts WHERE day_date = ? AND week_start = ?",
                    (day_date, week_start),
                ).fetchone()[0]
            else:
                count = con.execute(
                    "SELECT COUNT(*) FROM planned_workouts WHERE day_date = ?", (day_date,)
                ).fetchone()[0]

            if count == 0:
                return f"No planned sessions found for {day_date}."

            note_append = f"[{status.upper()}]" + (f": {reason}" if reason else "")

            if week_start:
                con.execute(
                    """
                    UPDATE planned_workouts
                    SET status     = ?,
                        notes      = CASE WHEN notes IS NULL THEN ? ELSE notes || ' | ' || ? END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE day_date = ? AND week_start = ?
                    """,
                    (status, note_append, note_append, day_date, week_start),
                )
            else:
                con.execute(
                    """
                    UPDATE planned_workouts
                    SET status     = ?,
                        notes      = CASE WHEN notes IS NULL THEN ? ELSE notes || ' | ' || ? END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE day_date = ?
                    """,
                    (status, note_append, note_append, day_date),
                )
            con.commit()

        msg = f"Updated {count} session(s) on {day_date} → '{status}'"
        return msg + (f" (reason: {reason})" if reason else "") + "."

    except Exception as exc:
        return f"Database error: {exc}"
