"""Proactive coaching jobs — run by PTB JobQueue on a schedule.

Each function has PTB's AsyncJobCallback signature:
    async def job(context: CallbackContext) -> None

DB / LLM calls are synchronous and run via asyncio.to_thread so the event
loop is never blocked.  Messages are sent to TELEGRAM_ALLOWED_USER_ID.
"""
import asyncio
import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from telegram.ext import CallbackContext

from src.config import config

logger = logging.getLogger(__name__)


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_llm() -> ChatOllama:
    return ChatOllama(
        model=config.MODEL_ID,
        temperature=0,
        base_url=config.OLLAMA_BASE_URL,
        num_ctx=8192,
        request_timeout=180,
    )


# ── Telegram send helper ──────────────────────────────────────────────────────

async def _send(bot, text: str) -> None:
    try:
        await bot.send_message(
            chat_id=config.TELEGRAM_ALLOWED_USER_ID,
            text=text,
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Failed to send proactive message")
        return

    try:
        from src.agent.graph import inject_bot_message_into_thread
        await inject_bot_message_into_thread(text)
    except Exception:
        logger.exception("Failed to store proactive message in thread")


# ── Job 1: Morning Briefing ───────────────────────────────────────────────────

def _gather_morning_data(today: str) -> str:
    from src.tools.readiness_tool import get_daily_readiness
    from src.tools.plan_tool import get_current_workout_plan

    readiness = get_daily_readiness.invoke({"date": today})
    plan = get_current_workout_plan.invoke({})
    return f"DATE: {today}\n\nREADINESS DATA:\n{readiness}\n\nTRAINING PLAN:\n{plan}"


async def morning_briefing_job(context: CallbackContext) -> None:
    """Send a morning readiness briefing with a workout adjustment recommendation."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = await asyncio.to_thread(_gather_morning_data, today)

        def _run():
            llm = _make_llm()
            system = (
                "You are an elite running coach delivering a concise morning briefing. "
                "Keep it under 200 words. Write in plain prose — no markdown, no bullet lists.\n"
                "Structure:\n"
                "1. Recovery status in one sentence (mention HRV, sleep hours, and TSB form score).\n"
                "2. Today's scheduled workout in one sentence.\n"
                "3. Green / Amber / Red light recommendation with a one-line reason.\n"
                "4. One actionable tip for the day.\n"
                "Use a warm, direct coaching tone."
            )
            return llm.invoke([SystemMessage(content=system), HumanMessage(content=data)])

        response = await asyncio.to_thread(_run)
        text = response.content if isinstance(response.content, str) else ""
        if text.strip():
            await _send(context.bot, f"🌅 <b>Morning Briefing</b>\n\n{text}")
    except Exception:
        logger.exception("morning_briefing_job failed")


# ── Job 2: Pre-Workout Reminder ───────────────────────────────────────────────

def _gather_preworkout_data(today: str) -> str | None:
    """Return a context string if a workout is planned today, else None."""
    from src.tools.readiness_tool import get_daily_readiness
    from src.tools._utils import db_ro

    with db_ro() as con:
        row = con.execute(
            """
            SELECT activity_type, workout_type, description,
                   target_distance_km, target_duration_min
            FROM planned_workouts
            WHERE day_date = %s AND deleted_at IS NULL AND status = 'planned'
            ORDER BY session_order
            LIMIT 1
            """,
            (today,),
        ).fetchone()

    if not row:
        return None

    readiness = get_daily_readiness.invoke({"date": today})
    dist = f"{row['target_distance_km']} km" if row["target_distance_km"] else ""
    dur = f"{row['target_duration_min']:.0f} min" if row["target_duration_min"] else ""
    return (
        f"DATE: {today}\n\n"
        f"TODAY'S PLANNED WORKOUT:\n"
        f"  Type: {row['workout_type']} {row['activity_type']} {dist} {dur}\n"
        f"  Description: {row['description']}\n\n"
        f"CURRENT READINESS:\n{readiness}"
    )


async def preworkout_reminder_job(context: CallbackContext) -> None:
    """Send a pre-workout briefing for today's scheduled session."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = await asyncio.to_thread(_gather_preworkout_data, today)
        if data is None:
            return  # No workout today — skip silently.

        def _run():
            llm = _make_llm()
            system = (
                "You are an elite coach sending a pre-workout briefing 30 minutes before training. "
                "Keep it under 150 words. Write in plain prose — no markdown, no bullet lists.\n"
                "Include:\n"
                "1. The session goal in one sentence.\n"
                "2. Target pace or load with a pacing-strategy tip (start conservative).\n"
                "3. One key focus cue for the session.\n"
                "4. A brief readiness note tied to today's HRV / sleep.\n"
                "End with one short encouraging sentence."
            )
            return llm.invoke([SystemMessage(content=system), HumanMessage(content=data)])

        response = await asyncio.to_thread(_run)
        text = response.content if isinstance(response.content, str) else ""
        if text.strip():
            await _send(context.bot, f"⚡ <b>Pre-Workout Briefing</b>\n\n{text}")
    except Exception:
        logger.exception("preworkout_reminder_job failed")


# ── Job 3: Daily Nutrition Plan ───────────────────────────────────────────────

def _gather_nutrition_data(today: str) -> str:
    from src.tools.nutrition_tool import get_nutrition_profile
    from src.tools._utils import db_ro

    profile = get_nutrition_profile.invoke({})

    with db_ro() as con:
        workout_row = con.execute(
            """
            SELECT activity_type, workout_type, intensity,
                   target_distance_km, target_duration_min
            FROM planned_workouts
            WHERE day_date = %s AND deleted_at IS NULL AND status = 'planned'
            ORDER BY session_order
            LIMIT 1
            """,
            (today,),
        ).fetchone()

    if workout_row:
        intensity = workout_row.get("intensity") or "moderate"
        wtype = f"{workout_row['workout_type']} {workout_row['activity_type']}"
        dist = f"{workout_row['target_distance_km']} km" if workout_row["target_distance_km"] else ""
        day_type = f"TRAINING DAY — {wtype} {dist} ({intensity} intensity)"
    else:
        day_type = "REST / RECOVERY DAY — no structured workout planned"

    return f"DATE: {today}\nDAY TYPE: {day_type}\n\n{profile}"


async def daily_nutrition_job(context: CallbackContext) -> None:
    """Send a personalized daily meal-plan recommendation."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = await asyncio.to_thread(_gather_nutrition_data, today)

        def _run():
            llm = _make_llm()
            system = (
                "You are a sports dietitian generating a practical daily nutrition plan. "
                "Keep it under 250 words. Write in plain prose — no markdown tables.\n"
                "Tailor calorie and macro targets to the day type (training vs rest). Structure:\n"
                "1. Total calorie target and macros (protein / carbs / fat) in one line each.\n"
                "2. Three-meal outline with one concrete food example per meal.\n"
                "3. Hydration target in litres.\n"
                "4. One specific nutrition tip for this type of training day."
            )
            return llm.invoke([SystemMessage(content=system), HumanMessage(content=data)])

        response = await asyncio.to_thread(_run)
        text = response.content if isinstance(response.content, str) else ""
        if text.strip():
            await _send(context.bot, f"🥗 <b>Daily Nutrition Plan</b>\n\n{text}")
    except Exception:
        logger.exception("daily_nutrition_job failed")


# ── Job 4: Evening Summary ────────────────────────────────────────────────────

def _gather_evening_data(today: str) -> str:
    from src.tools.readiness_tool import get_daily_readiness
    from src.tools._utils import db_ro

    with db_ro() as con:
        completed = con.execute(
            """
            SELECT activity_type, duration_min, distance_km,
                   avg_heart_rate_bpm, avg_speed_kmh, training_stress_score, rpe, notes
            FROM workouts
            WHERE start_date::date = %s::date
            ORDER BY start_date
            """,
            (today,),
        ).fetchall()

        planned = con.execute(
            """
            SELECT activity_type, workout_type, description,
                   target_distance_km, target_duration_min, intensity, status
            FROM planned_workouts
            WHERE day_date = %s AND deleted_at IS NULL
            ORDER BY session_order
            """,
            (today,),
        ).fetchall()

    readiness = get_daily_readiness.invoke({"date": today})

    lines = [f"DATE: {today}\n\nCOMPLETED WORKOUTS:"]
    for w in completed:
        dist = f"{w['distance_km']:.2f} km" if w["distance_km"] else "—"
        dur = f"{w['duration_min']:.0f} min" if w["duration_min"] else "—"
        hr = f"{w['avg_heart_rate_bpm']:.0f} bpm" if w["avg_heart_rate_bpm"] else "—"
        pace = (
            f"{60 / w['avg_speed_kmh']:.2f} min/km" if w["avg_speed_kmh"] else "—"
        )
        lines.append(
            f"  {w['activity_type']}: {dist} / {dur} | avg HR {hr} | pace {pace}"
            + (f" | RPE {w['rpe']}" if w["rpe"] else "")
        )
    if not completed:
        lines.append("  (none recorded)")

    lines.append("\nPLANNED:")
    for p in planned:
        lines.append(
            f"  {p['workout_type']} {p['activity_type']} — status: {p['status']}"
        )
    if not planned:
        lines.append("  (rest day — nothing scheduled)")

    lines.append(f"\nREADINESS:\n{readiness}")
    return "\n".join(lines)


async def evening_summary_job(context: CallbackContext) -> None:
    """Send an end-of-day training summary aligned with the plan and goals."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = await asyncio.to_thread(_gather_evening_data, today)

        def _run():
            llm = _make_llm()
            system = (
                "You are a running coach delivering an end-of-day training summary. "
                "Keep it under 200 words. Write in plain prose — no markdown.\n"
                "Cover:\n"
                "1. What was done today vs what was planned (one paragraph).\n"
                "2. How today aligns with the weekly plan and long-term goals.\n"
                "3. Recovery recommendation for tonight (sleep priority, any nutrition note).\n"
                "4. One-sentence preview of tomorrow.\n"
                "Be direct, supportive, and factual."
            )
            return llm.invoke([SystemMessage(content=system), HumanMessage(content=data)])

        response = await asyncio.to_thread(_run)
        text = response.content if isinstance(response.content, str) else ""
        if text.strip():
            await _send(context.bot, f"🌙 <b>Evening Summary</b>\n\n{text}")
    except Exception:
        logger.exception("evening_summary_job failed")


# ── Job 5: Weekend Weekly Review ──────────────────────────────────────────────

async def weekend_review_job(context: CallbackContext) -> None:
    """Saturday-night job: full weekly review + next-week plan generation."""
    from src.scheduler.weekly_supervisor import run_weekly_review
    try:
        await run_weekly_review(context.bot)
    except Exception:
        logger.exception("weekend_review_job failed")
