import logging
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools._utils import db_ro, db_rw

logger = logging.getLogger(__name__)


class WorkoutSession(BaseModel):
    """One session inside a weekly training plan."""
    day_date: str = Field(..., description="Session date in YYYY-MM-DD format")
    activity_type: Literal["running", "strength", "rest", "cross_training"]
    workout_type: Literal[
        "easy", "tempo", "interval", "long_run", "recovery",
        "strength", "rest", "assessment",
    ]
    description: str = Field(
        ..., description="Full session description including exact target paces (min/km) or loads (kg)"
    )
    intensity: Literal["easy", "moderate", "hard", "rest"]
    target_distance_km: Optional[float] = Field(None, description="Target distance in km (running sessions)")
    target_duration_min: Optional[float] = Field(None, description="Target duration in minutes")
    phase: Optional[Literal[
        "onboarding", "base", "build", "peak", "race", "recovery", "return_to_run"
    ]] = None
    is_assessment: int = Field(0, description="1 for time trials or strength tests, 0 otherwise")
    notes: Optional[str] = Field(None, description="Coach rationale or additional context")


class DaySession(BaseModel):
    """One session when replacing a single day (day_date is supplied at the tool level)."""
    activity_type: Literal["running", "strength", "rest", "cross_training"]
    workout_type: Literal[
        "easy", "tempo", "interval", "long_run", "recovery",
        "strength", "rest", "assessment",
    ]
    description: str = Field(
        ..., description="Full session description including exact target paces (min/km) or loads (kg)"
    )
    intensity: Literal["easy", "moderate", "hard", "rest"]
    target_distance_km: Optional[float] = None
    target_duration_min: Optional[float] = None
    phase: Optional[Literal[
        "onboarding", "base", "build", "peak", "race", "recovery", "return_to_run"
    ]] = None
    is_assessment: int = Field(0, description="1 for time trials or strength tests, 0 otherwise")
    notes: Optional[str] = None


@tool
def save_workout_plan(week_start: str, sessions: list[WorkoutSession]) -> str:
    """
    Save a weekly training plan to the database. Replaces any existing plan for that week.
    Call this after generating a workout plan for the athlete.

    Args:
        week_start: The Sunday that starts the plan week in 'YYYY-MM-DD' format.
        sessions: List of WorkoutSession objects — each field is strictly typed (see schema).
    """
    try:
        with db_rw() as con:
            # Soft-delete existing sessions for this week instead of hard-deleting,
            # so an LLM hallucinating the wrong week_start never destroys history.
            con.execute(
                "UPDATE planned_workouts SET deleted_at = NOW() "
                "WHERE week_start = %s AND deleted_at IS NULL",
                (week_start,),
            )
            for order, s in enumerate(sessions, start=1):
                con.execute(
                    """
                    INSERT INTO planned_workouts
                        (week_start, day_date, session_order, activity_type, workout_type,
                         description, target_distance_km, target_duration_min,
                         intensity, phase, is_assessment, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        week_start,
                        s.day_date,
                        order,
                        s.activity_type,
                        s.workout_type,
                        s.description,
                        s.target_distance_km,
                        s.target_duration_min,
                        s.intensity,
                        s.phase,
                        s.is_assessment,
                        s.notes,
                    ),
                )
            con.commit()
        assessment_count = sum(1 for s in sessions if s.is_assessment)
        summary = f"Saved {len(sessions)} sessions for the week of {week_start}."
        if assessment_count:
            summary += f" ({assessment_count} assessment session(s) included.)"
        return summary
    except Exception as exc:
        logger.exception("Tool error: %s", exc)
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
                WHERE week_start = %s AND deleted_at IS NULL
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
                    WHERE week_start > %s AND deleted_at IS NULL
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
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"


@tool
def replace_day_in_plan(week_start: str, day_date: str, sessions: list[DaySession]) -> str:
    """
    Replace all sessions for a single day within an existing weekly plan.
    Use this to adjust, swap, or add a session for one day without
    rewriting the entire week.

    Args:
        week_start: The Sunday that starts the plan week in 'YYYY-MM-DD' format.
        day_date: The specific day to update in 'YYYY-MM-DD' format.
        sessions: List of DaySession objects (day_date is supplied above, not per-session).
    """
    try:
        with db_rw() as con:
            con.execute(
                "UPDATE planned_workouts SET deleted_at = NOW() "
                "WHERE week_start = %s AND day_date = %s AND deleted_at IS NULL",
                (week_start, day_date),
            )
            for order, s in enumerate(sessions, start=1):
                con.execute(
                    """
                    INSERT INTO planned_workouts
                        (week_start, day_date, session_order, activity_type, workout_type,
                         description, target_distance_km, target_duration_min,
                         intensity, phase, is_assessment, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        week_start,
                        day_date,
                        order,
                        s.activity_type,
                        s.workout_type,
                        s.description,
                        s.target_distance_km,
                        s.target_duration_min,
                        s.intensity,
                        s.phase,
                        s.is_assessment,
                        s.notes,
                    ),
                )
            con.commit()
        return f"Updated {len(sessions)} session(s) for {day_date} in the week of {week_start}."
    except Exception as exc:
        logger.exception("Tool error: %s", exc)
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
                    "SELECT COUNT(*) FROM planned_workouts WHERE day_date = %s AND week_start = %s",
                    (day_date, week_start),
                ).fetchone()["count"]
            else:
                count = con.execute(
                    "SELECT COUNT(*) FROM planned_workouts WHERE day_date = %s", (day_date,)
                ).fetchone()["count"]

            if count == 0:
                return f"No planned sessions found for {day_date}."

            note_append = f"[{status.upper()}]" + (f": {reason}" if reason else "")

            if week_start:
                con.execute(
                    """
                    UPDATE planned_workouts
                    SET status     = %s,
                        notes      = CASE WHEN notes IS NULL THEN %s ELSE notes || ' | ' || %s END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE day_date = %s AND week_start = %s
                    """,
                    (status, note_append, note_append, day_date, week_start),
                )
            else:
                con.execute(
                    """
                    UPDATE planned_workouts
                    SET status     = %s,
                        notes      = CASE WHEN notes IS NULL THEN %s ELSE notes || ' | ' || %s END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE day_date = %s
                    """,
                    (status, note_append, note_append, day_date),
                )
            con.commit()

        msg = f"Updated {count} session(s) on {day_date} → '{status}'"
        return msg + (f" (reason: {reason})" if reason else "") + "."

    except Exception as exc:
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"
