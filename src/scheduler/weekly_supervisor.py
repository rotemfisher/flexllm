"""Weekly supervisor agent — runs automatically every Saturday night.

Pipeline:
1. Gather full week data (workouts completed, plan adherence, daily health).
2. Call LLM for a written review narrative.
3. Call LLM in JSON mode to produce the next-week training plan.
4. Write the plan into planned_workouts (soft-deletes the previous one).
5. Send an executive summary to Telegram with approve/adjust inline buttons.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from src.config import config

logger = logging.getLogger(__name__)

WEEKLY_PLAN_APPROVED = "weekly_plan:approved"
WEEKLY_PLAN_ADJUST   = "weekly_plan:adjust"

_TG_MAX = 3900


# ── LLM factories ─────────────────────────────────────────────────────────────

def _review_llm() -> ChatOllama:
    return ChatOllama(
        model=config.MODEL_ID,
        temperature=0,
        base_url=config.OLLAMA_BASE_URL,
        num_ctx=16384,
        request_timeout=300,
    )


def _plan_llm() -> ChatOllama:
    return ChatOllama(
        model=config.MODEL_ID,
        temperature=0,
        format="json",
        base_url=config.OLLAMA_BASE_URL,
        num_ctx=16384,
        request_timeout=300,
    )


# ── Data gathering ─────────────────────────────────────────────────────────────

def _gather_week_data() -> dict:
    from src.tools._utils import db_ro

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    # Sunday-based week start — same convention as plan_tool.py
    week_start = (now - timedelta(days=(now.weekday() + 1) % 7)).strftime("%Y-%m-%d")

    with db_ro() as con:
        workouts = con.execute(
            """
            SELECT activity_type, start_date, duration_min, distance_km,
                   avg_heart_rate_bpm, avg_speed_kmh, training_stress_score, rpe, notes
            FROM workouts
            WHERE start_date >= %s AND start_date < %s
            ORDER BY start_date
            """,
            (week_start, today),
        ).fetchall()

        planned = con.execute(
            """
            SELECT day_date, activity_type, workout_type, description,
                   target_distance_km, target_duration_min, intensity, phase, status
            FROM planned_workouts
            WHERE week_start = %s AND deleted_at IS NULL
            ORDER BY day_date, session_order
            """,
            (week_start,),
        ).fetchall()

        health_rows = con.execute(
            """
            SELECT date, resting_heart_rate_bpm, hrv_sdnn_ms,
                   sleep_total_min, sleep_deep_min, atl, ctl, tsb
            FROM daily_health
            WHERE date >= %s AND date <= %s
            ORDER BY date
            """,
            (week_start, today),
        ).fetchall()

        profile = con.execute(
            "SELECT fitness_level, current_goal FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "week_start": week_start,
        "today": today,
        "workouts": [dict(r) for r in workouts],
        "planned":  [dict(r) for r in planned],
        "health":   [dict(r) for r in health_rows],
        "profile":  dict(profile) if profile else {},
    }


def _format_week_context(data: dict) -> str:
    lines = [
        f"WEEK: {data['week_start']} → {data['today']}",
        f"ATHLETE: fitness_level={data['profile'].get('fitness_level', '?')} "
        f"| goal={data['profile'].get('current_goal', '?')}",
        "",
        "=== COMPLETED WORKOUTS ===",
    ]
    for w in data["workouts"]:
        date   = str(w["start_date"])[:10]
        dist   = f"{w['distance_km']:.2f} km" if w["distance_km"] else "—"
        dur    = f"{w['duration_min']:.0f} min" if w["duration_min"] else "—"
        hr     = f"{w['avg_heart_rate_bpm']:.0f} bpm" if w["avg_heart_rate_bpm"] else "—"
        pace   = f"{60 / w['avg_speed_kmh']:.2f} min/km" if w["avg_speed_kmh"] else "—"
        tss    = f"TSS={w['training_stress_score']:.0f}" if w["training_stress_score"] else ""
        rpe    = f"RPE={w['rpe']}" if w["rpe"] else ""
        lines.append(
            f"  {date} {w['activity_type']}: {dist}/{dur} pace={pace} HR={hr} {tss} {rpe}"
        )
    if not data["workouts"]:
        lines.append("  (none recorded this week)")

    lines += ["", "=== PLAN ADHERENCE ==="]
    for p in data["planned"]:
        lines.append(
            f"  {p['day_date']}: {p['workout_type']} {p['activity_type']} — {p['status']}"
        )
    if not data["planned"]:
        lines.append("  (no plan was set for this week)")

    lines += ["", "=== DAILY HEALTH METRICS ==="]
    for h in data["health"]:
        sleep_h = round((h["sleep_total_min"] or 0) / 60, 1)
        lines.append(
            f"  {h['date']}: RHR={h['resting_heart_rate_bpm'] or '?'} bpm "
            f"HRV={h['hrv_sdnn_ms'] or '?'} ms "
            f"Sleep={sleep_h}h "
            f"ATL={h['atl'] or '?'} CTL={h['ctl'] or '?'} TSB={h['tsb'] or '?'}"
        )
    if not data["health"]:
        lines.append("  (no health data)")

    return "\n".join(lines)


# ── Next-week date calculation ────────────────────────────────────────────────

def _next_week_start(today: str) -> str:
    """Return the Sunday immediately following today's week."""
    d = datetime.fromisoformat(today)
    # Python weekday: Monday=0 … Saturday=5, Sunday=6
    days_to_next_sunday = (6 - d.weekday()) % 7
    if days_to_next_sunday == 0:
        days_to_next_sunday = 7  # today is already Sunday → skip to next Sunday
    return (d + timedelta(days=days_to_next_sunday)).strftime("%Y-%m-%d")


# ── LLM calls (sync — run in thread) ─────────────────────────────────────────

def _call_review_llm(context_text: str) -> str:
    llm = _review_llm()
    system = (
        "You are an expert running coach writing a professional weekly training review. "
        "Keep it under 300 words. Write in clear, professional prose — no markdown headers.\n"
        "Cover:\n"
        "1. Training volume and intensity summary for the week.\n"
        "2. Plan adherence — what was hit, what was missed and why.\n"
        "3. Key recovery metrics: average HRV trend and sleep.\n"
        "4. The biggest win of the week.\n"
        "5. The main challenge or area to address.\n"
        "6. Tone and focus recommendations for the coming week."
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=context_text)])
    return resp.content if isinstance(resp.content, str) else ""


def _call_plan_llm(context_text: str, review_text: str, next_week_start: str) -> list[dict]:
    """Return a list of session dicts for next week, or [] on parse failure."""
    llm = _plan_llm()
    system = (
        f"You are an expert running coach. Generate a 7-day training plan for the week starting "
        f"on {next_week_start} (Sunday). Apply progressive overload and include at least one rest day.\n\n"
        "Output ONLY a JSON object with a single key 'sessions' containing a list of 7 objects. "
        "Each object must have these exact keys:\n"
        "  day_date           : 'YYYY-MM-DD'\n"
        "  activity_type      : 'running' | 'strength' | 'rest' | 'cross_training'\n"
        "  workout_type       : 'easy' | 'tempo' | 'interval' | 'long_run' | 'recovery' "
        "| 'strength' | 'rest' | 'assessment'\n"
        "  description        : full session description with exact target paces (min/km) or loads (kg)\n"
        "  intensity          : 'easy' | 'moderate' | 'hard' | 'rest'\n"
        "  target_distance_km : number or null\n"
        "  target_duration_min: number or null\n"
        "  phase              : 'base' | 'build' | 'peak' | 'race' | 'recovery' | null\n"
        "  is_assessment      : 0 or 1\n"
        "  notes              : string or null"
    )
    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(
            content=f"Weekly data:\n{context_text}\n\nWeek review summary:\n{review_text}"
        ),
    ])
    raw = resp.content if isinstance(resp.content, str) else "{}"
    try:
        return json.loads(raw).get("sessions", [])
    except json.JSONDecodeError:
        logger.warning("weekly_supervisor: plan JSON parse failed — raw:\n%s", raw[:500])
        return []


# ── DB write ──────────────────────────────────────────────────────────────────

def _save_plan(week_start: str, sessions: list[dict]) -> str:
    from src.tools._utils import db_rw

    if not sessions:
        return "⚠️ No valid plan generated — nothing written to DB."
    try:
        with db_rw() as con:
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
                        s.get("day_date"),
                        order,
                        s.get("activity_type", "rest"),
                        s.get("workout_type", "rest"),
                        s.get("description", ""),
                        s.get("target_distance_km"),
                        s.get("target_duration_min"),
                        s.get("intensity", "easy"),
                        s.get("phase"),
                        int(s.get("is_assessment", 0)),
                        s.get("notes"),
                    ),
                )
            con.commit()
        return f"✅ {len(sessions)}-session plan for week of {week_start} saved to DB."
    except Exception as exc:
        logger.exception("_save_plan failed")
        return f"❌ DB write failed: {exc}"


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_weekly_review(bot) -> None:
    """Run the full weekly supervisor pipeline and send the result to Telegram."""
    try:
        data = await asyncio.to_thread(_gather_week_data)
        context_text = _format_week_context(data)
        next_week = _next_week_start(data["today"])

        review_text = await asyncio.to_thread(_call_review_llm, context_text)
        sessions    = await asyncio.to_thread(
            _call_plan_llm, context_text, review_text, next_week
        )
        save_status = await asyncio.to_thread(_save_plan, next_week, sessions)

        # Build plan preview for Telegram
        session_lines = []
        for s in sessions:
            dist = f" {s['target_distance_km']} km" if s.get("target_distance_km") else ""
            session_lines.append(
                f"  {s.get('day_date', '?')}: "
                f"{s.get('workout_type', '?').title()} {s.get('activity_type', '?')}{dist}"
            )
        plan_preview = "\n".join(session_lines) if session_lines else "  (no sessions)"

        msg = (
            f"📊 <b>Weekly Review — {data['week_start']}</b>\n\n"
            f"{review_text}\n\n"
            f"<b>Next Week ({next_week}):</b>\n{plan_preview}\n\n"
            f"{save_status}\n\n"
            "Message me to adjust any session before your first workout."
        )
        if len(msg) > _TG_MAX:
            msg = msg[:_TG_MAX] + "…"

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Looks good!", callback_data=WEEKLY_PLAN_APPROVED),
            InlineKeyboardButton("✏️ Adjust in chat", callback_data=WEEKLY_PLAN_ADJUST),
        ]])

        await bot.send_message(
            chat_id=config.TELEGRAM_ALLOWED_USER_ID,
            text=msg,
            parse_mode="HTML",
            reply_markup=keyboard,
        )
    except Exception:
        logger.exception("run_weekly_review pipeline failed")
        try:
            await bot.send_message(
                chat_id=config.TELEGRAM_ALLOWED_USER_ID,
                text="⚠️ Weekly review failed — check the logs.",
            )
        except Exception:
            pass
