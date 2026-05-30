from langchain_core.tools import tool

from src.tools._utils import db_ro


@tool
def get_active_injuries() -> str:
    """
    Check the athlete's current active injuries or niggles.
    ALWAYS call this tool before recommending a high-intensity workout.
    """
    try:
        with db_ro() as con:
            injuries = con.execute(
                """
                SELECT id, body_part, side, injury_type, severity, status, pain_scale, pain_context
                FROM injuries
                WHERE status IN ('active', 'recovering')
                ORDER BY onset_date DESC
                """
            ).fetchall()

        if not injuries:
            return "Great news: The athlete currently has NO active injuries. Cleared for standard training."

        report = "WARNING: The athlete has the following active/recovering injuries:\n\n"
        for inj in injuries:
            report += (
                f"- [{inj['id']}] {inj['severity'].title()} {inj['side']} {inj['body_part']} "
                f"({inj['injury_type'] or 'unclassified'})\n"
                f"  Status: {inj['status'].title()}\n"
                f"  Pain Scale: {inj['pain_scale'] or 'N/A'}/10 "
                f"(Context: {inj['pain_context'] or 'N/A'})\n\n"
            )

        report += (
            "Physio Directive: Adjust the training plan to accommodate these injuries. "
            "Remove contraindicated movements. Use get_injury_recovery_trend(injury_id) "
            "to check recovery trajectory before returning to training."
        )
        return report

    except Exception as exc:
        return f"Database error: {exc}"


@tool
def get_injury_recovery_trend(injury_id: int, days: int = 14) -> str:
    """
    Get the pain progression history for an active injury.
    Use this to decide when the athlete is ready to return to training.

    A return-to-train signal fires when pain has been <=2 for 3 or more consecutive days.

    Args:
        injury_id: ID of the injury (visible in get_active_injuries output).
        days: how many days of check-ins to review (default 14).
    """
    days = min(max(days, 3), 90)
    try:
        with db_ro() as con:
            injury = con.execute(
                "SELECT body_part, side, injury_type, severity, status, onset_date FROM injuries WHERE id = ?",
                (injury_id,),
            ).fetchone()
            if not injury:
                return f"Error: injury ID {injury_id} not found. Use get_active_injuries to find IDs."

            checks = con.execute(
                """
                SELECT check_date, pain_scale, pain_context, notes
                FROM injury_checks
                WHERE injury_id = ? AND check_date >= date('now', ?)
                ORDER BY check_date ASC
                """,
                (injury_id, f"-{days} days"),
            ).fetchall()

        label = (
            f"{injury['severity'].title()} {injury['side']} {injury['body_part']}"
            + (f" ({injury['injury_type']})" if injury["injury_type"] else "")
        )
        lines = [f"--- Recovery Trend: {label} | Status: {injury['status'].title()} ---\n"]

        if not checks:
            return (
                f"{lines[0]}\nNo check-ins in the last {days} days. "
                f"Use log_injury_checkin({injury_id}, ...) to start tracking daily pain."
            )

        # Filter out any NULL pain_scale values before numeric operations.
        pain_values = [c["pain_scale"] for c in checks if c["pain_scale"] is not None]
        for c in checks:
            scale = c["pain_scale"] if c["pain_scale"] is not None else 0
            filled = "#" * scale
            empty  = "-" * (10 - scale)
            lines.append(
                f"  {c['check_date']}  {filled}{empty}  {scale}/10  ({c['pain_context']})"
                + (f"  — {c['notes']}" if c["notes"] else "")
            )

        # Trend signal
        lines.append("")
        if len(pain_values) >= 4:
            first_half  = pain_values[: len(pain_values) // 2]
            second_half = pain_values[len(pain_values) // 2 :]
            delta = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))
            if delta < -1.0:
                lines.append("Trend: IMPROVING (decreasing)")
            elif delta > 1.0:
                lines.append("Trend: WORSENING (increasing) — reduce training load immediately")
            else:
                lines.append("Trend: STABLE")

        # Return-to-train decision
        recent = pain_values[-3:]
        if len(recent) == 3 and all(p <= 2 for p in recent):
            lines.append(
                "\nRETURN-TO-TRAIN CLEARED: Pain <=2 for 3 consecutive days.\n"
                "   Protocol:\n"
                "   Phase 1 (wk 1): 30% of previous volume, easy only, no quality sessions.\n"
                "   Phase 2 (wk 2): 50% volume if pain stays <=2. Reintroduce moderate work.\n"
                "   Phase 3 (wk 3+): 70% volume. Add one quality session if pain-free.\n"
                "   Use save_workout_plan with phase='return_to_run' for Phase 1."
            )
        elif len(recent) >= 2 and all(p <= 4 for p in recent):
            lines.append(
                "\nBORDERLINE: Pain controlled but not cleared. "
                "2-3 more pain-free days needed before return."
            )
        else:
            latest = pain_values[-1] if pain_values else None
            if latest is not None and latest >= 5:
                lines.append("\nNOT READY: Pain still significant. Maintain full recovery plan.")

        return "\n".join(lines)

    except Exception as exc:
        return f"Database error: {exc}"
