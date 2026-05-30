from datetime import datetime, timezone

from langchain_core.tools import tool

from src.tools._utils import db_rw


@tool
def log_injury(
    body_part: str,
    side: str,
    severity: str,
    pain_scale: int,
    pain_context: str,
    onset_date: str | None = None,
    injury_type: str | None = None,
    cause: str | None = None,
    notes: str | None = None,
) -> str:
    """
    Record a new injury or niggle in the athlete's injury log.
    Call this when the athlete reports a new pain, soreness, or injury for the first time.
    For follow-up updates on an existing injury, use log_injury_checkin instead.

    Args:
        body_part: anatomical location, e.g. 'knee', 'achilles', 'shin', 'hip', 'foot', 'hamstring', 'calf'
        side: 'left' | 'right' | 'bilateral' | 'central'
        severity: 'mild' | 'moderate' | 'severe'
        pain_scale: 0–10 (0 = no pain, 10 = worst imaginable)
        pain_context: when pain occurs — 'workout' | 'recovery' | 'rest' | 'both'
        onset_date: 'YYYY-MM-DD' when symptoms first appeared. Defaults to today.
        injury_type: clinical label e.g. 'ITBS', 'plantar fasciitis', 'stress fracture', 'muscle strain'
        cause: 'overuse' | 'acute trauma' | 'overtraining' | 'unknown'
        notes: any additional context the athlete described
    """
    _VALID = {
        "side":          ("left", "right", "bilateral", "central"),
        "severity":      ("mild", "moderate", "severe"),
        "pain_context":  ("workout", "recovery", "rest", "both"),
    }
    _VALUES = {"side": side, "severity": severity, "pain_context": pain_context}
    for field, allowed in _VALID.items():
        if _VALUES[field] not in allowed:
            return f"Error: {field} must be one of {allowed}. Got '{_VALUES[field]}'."
    if not (0 <= pain_scale <= 10):
        return "Error: pain_scale must be between 0 and 10."

    if not onset_date:
        onset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        with db_rw() as con:
            cur = con.execute(
                """
                INSERT INTO injuries
                    (onset_date, body_part, side, injury_type, cause,
                     severity, status, pain_scale, pain_context, notes)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (onset_date, body_part, side, injury_type, cause,
                 severity, pain_scale, pain_context, notes),
            )
            con.commit()
            injury_id = cur.lastrowid
        return (
            f"Injury logged (ID {injury_id}): {severity} {side} {body_part}"
            + (f" ({injury_type})" if injury_type else "")
            + f" — pain {pain_scale}/10 during {pain_context}, onset {onset_date}. "
            f"Use injury ID {injury_id} for daily check-ins with log_injury_checkin."
        )
    except Exception as exc:
        return f"Database error: {exc}"


@tool
def log_injury_checkin(
    injury_id: int,
    pain_scale: int,
    pain_context: str,
    notes: str | None = None,
) -> str:
    """
    Log a daily pain check-in for an existing injury to track its progression.
    Call this whenever the athlete gives an update on how an injury is feeling today.
    Use get_active_injuries first to find the correct injury ID.

    Args:
        injury_id: ID of the injury (returned by log_injury or visible in get_active_injuries).
        pain_scale: 0–10 (0 = no pain, 10 = worst imaginable)
        pain_context: 'workout' | 'recovery' | 'rest' | 'both'
        notes: athlete's description of how it felt today
    """
    if not (0 <= pain_scale <= 10):
        return "Error: pain_scale must be between 0 and 10."
    if pain_context not in ("workout", "recovery", "rest", "both"):
        return "Error: pain_context must be 'workout', 'recovery', 'rest', or 'both'."

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with db_rw() as con:
            row = con.execute(
                "SELECT body_part, side, pain_scale FROM injuries WHERE id = ?", (injury_id,)
            ).fetchone()
            if not row:
                return f"Error: injury ID {injury_id} not found. Use get_active_injuries to see current IDs."

            prev_pain = row[2]
            con.execute(
                """
                INSERT INTO injury_checks (injury_id, check_date, pain_scale, pain_context, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (injury_id, today, pain_scale, pain_context, notes),
            )
            con.execute(
                """
                UPDATE injuries
                SET pain_scale = ?, pain_context = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (pain_scale, pain_context, injury_id),
            )
            con.commit()

        trend = ""
        if prev_pain is not None:
            delta = pain_scale - prev_pain
            if delta < 0:
                trend = f" (improving: was {prev_pain}/10)"
            elif delta > 0:
                trend = f" (worsening: was {prev_pain}/10)"
            else:
                trend = f" (unchanged from {prev_pain}/10)"

        return (
            f"Check-in recorded for {row[1]} {row[0]} (ID {injury_id}): "
            f"pain {pain_scale}/10 ({pain_context}) on {today}{trend}."
        )
    except Exception as exc:
        return f"Database error: {exc}"


@tool
def resolve_injury(injury_id: int, notes: str | None = None) -> str:
    """
    Mark an active injury as resolved/recovered.
    Call this when the athlete reports they are fully pain-free and the injury has healed.

    Args:
        injury_id: ID of the injury (from get_active_injuries).
        notes: optional closing note, e.g. 'fully pain-free after 2 weeks of rest'.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with db_rw() as con:
            row = con.execute(
                "SELECT body_part, side, status FROM injuries WHERE id = ?", (injury_id,)
            ).fetchone()
            if not row:
                return f"Error: injury ID {injury_id} not found."
            if row[2] == "resolved":
                return f"Injury ID {injury_id} is already marked as resolved."

            con.execute(
                """
                UPDATE injuries
                SET status = 'resolved', resolved_date = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (today, injury_id),
            )
            if notes:
                con.execute(
                    """
                    INSERT INTO injury_checks (injury_id, check_date, pain_scale, pain_context, notes)
                    VALUES (?, ?, 0, 'rest', ?)
                    """,
                    (injury_id, today, f"Resolved: {notes}"),
                )
            con.commit()
        return f"Injury resolved: {row[1]} {row[0]} (ID {injury_id}) marked as recovered on {today}."
    except Exception as exc:
        return f"Database error: {exc}"
