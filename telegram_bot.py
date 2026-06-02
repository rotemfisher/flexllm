"""FlexLLM Coach — Telegram bot entry point.

Run with:
    python telegram_bot.py

Uses python-telegram-bot v20+ (async, polling mode — no webhook needed).
ConversationHandler drives onboarding via inline keyboards and free-text
replies; once complete it hands off to the coaching chat backed by the same
LangGraph multi-agent graph used by the old Chainlit frontend.

A minimal FastAPI health server runs on port 8000 in a background daemon
thread so Docker's healthcheck and the watcher's depends_on still work.
"""
import asyncio
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import ExitStack
import atexit

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import AIMessageChunk, HumanMessage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from src.api.app import setup_logging
from src.api.routes import health as health_routes
from src.agent.coach_agent import build_coach_graph, get_athlete_context
from src.config import config
from src.tracing import setup_tracing
from src import api as _api_pkg

logger = logging.getLogger(__name__)

# ── ConversationHandler states ────────────────────────────────────────────────
(
    ONBOARDING_NAME,
    ONBOARDING_DOB,
    ONBOARDING_WEIGHT,
    ONBOARDING_GOAL,
    ONBOARDING_LEVEL,
    ONBOARDING_MEDICAL,
    CHAT,
) = range(7)

_AGENT_NODES = frozenset(
    {"trainer", "physiotherapist", "recovery_coach", "dietitian", "psychologist"}
)

# ── One graph shared across all Telegram users ────────────────────────────────
_exit_stack = ExitStack()
_graph = _exit_stack.enter_context(build_coach_graph())
atexit.register(_exit_stack.close)

# Expose graph to the /health/ready endpoint
_api_pkg.dependencies.compiled_graph = _graph

# ── Schema migration (same columns as the old Chainlit flow) ──────────────────
_MIGRATION_STMTS = [
    "ALTER TABLE athlete_profile ADD COLUMN name TEXT",
    "ALTER TABLE athlete_profile ADD COLUMN current_weight_kg REAL",
    "ALTER TABLE athlete_profile ADD COLUMN medical_conditions TEXT",
]
try:
    _mig_con = sqlite3.connect(config.DB_PATH)
    for _stmt in _MIGRATION_STMTS:
        try:
            _mig_con.execute(_stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    _mig_con.commit()
    _mig_con.close()
except Exception as _exc:
    logger.warning("Schema migration: %s", _exc)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _onboarding_complete() -> bool:
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        row = con.execute(
            "SELECT onboarding_complete FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
        return bool(row and row[0] == 1)
    except Exception:
        return False


def _save_profile(*, name, dob, weight, goal, fitness_level, medical) -> None:
    con = sqlite3.connect(config.DB_PATH)
    con.execute(
        """
        INSERT INTO athlete_profile
            (name, date_of_birth, current_goal, fitness_level,
             current_weight_kg, medical_conditions, onboarding_complete,
             updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, strftime('%Y-%m-%d %H:%M:%S', 'now'))
        """,
        (name, dob, goal, fitness_level, weight, medical),
    )
    con.commit()
    con.close()


# ── Graph helpers ─────────────────────────────────────────────────────────────

def _run_config(chat_id: int) -> dict:
    thread_id = str(chat_id)
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"coaching-{thread_id[:8]}",
        "tags": [config.ENVIRONMENT, "telegram", config.MODEL_ID],
        "metadata": {"source": "telegram", "thread_id": thread_id},
    }


async def _stream_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
) -> None:
    """Send user_text to the LangGraph coach and stream chunks back as a Telegram message."""
    chat_id = update.effective_chat.id
    athlete_ctx = context.chat_data.get("athlete_ctx", "")
    run_cfg = _run_config(chat_id)

    placeholder = await update.effective_chat.send_message("⏳ Thinking…")

    buffer = ""
    current_agent: str | None = None
    last_edit_at = 0.0
    MIN_EDIT_INTERVAL = 2.5  # seconds — Telegram allows ~1 edit/s; be conservative

    async for chunk, metadata in _graph.astream(
        {"messages": [HumanMessage(content=user_text)], "athlete_context": athlete_ctx},
        config=run_cfg,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        if node in _AGENT_NODES and isinstance(chunk, AIMessageChunk):
            if node != current_agent:
                if current_agent is not None:
                    buffer += "\n\n"
                label = node.replace("_", " ").title()
                buffer += f"*[{label}]*\n"
                current_agent = node
            if isinstance(chunk.content, str) and chunk.content:
                buffer += chunk.content

            now = asyncio.get_event_loop().time()
            if buffer and now - last_edit_at >= MIN_EDIT_INTERVAL:
                try:
                    await placeholder.edit_text(buffer + " ▌")
                    last_edit_at = now
                except Exception:
                    pass

    if not buffer:
        buffer = "(No response from coach)"

    # Final edit: try Markdown, fall back to plain text
    try:
        await placeholder.edit_text(buffer, parse_mode="Markdown")
    except Exception:
        try:
            await placeholder.edit_text(buffer)
        except Exception:
            pass


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if _onboarding_complete():
        athlete_ctx = get_athlete_context()
        context.chat_data["athlete_ctx"] = athlete_ctx
        await update.message.reply_text(
            "*FlexLLM Coach is ready.*\n\n"
            "Ask me anything about your training, nutrition, recovery, or wellbeing.\n\n"
            "---\n"
            f"{athlete_ctx}",
            parse_mode="Markdown",
        )
        return CHAT

    await update.message.reply_text(
        "*Welcome to FlexLLM Coach!*\n\n"
        "Before we start I need a few details to personalise your experience. "
        "This only happens once and takes about 2 minutes.\n\n"
        "What is your name?",
        parse_mode="Markdown",
    )
    return ONBOARDING_NAME


# ── Onboarding steps ──────────────────────────────────────────────────────────

async def onboarding_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = update.message.text.strip() or "Athlete"
    context.chat_data["ob_name"] = name
    await update.message.reply_text(
        f"Hi *{name}*\\! What is your date of birth?\n_\\(YYYY\\-MM\\-DD, e\\.g\\. 1995\\-03\\-21\\)_",
        parse_mode="MarkdownV2",
    )
    return ONBOARDING_DOB


async def onboarding_dob(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.chat_data["ob_dob"] = update.message.text.strip() or "1990-01-01"
    await update.message.reply_text(
        "What is your current weight in kilograms?\n_(e.g. 78.5)_",
        parse_mode="Markdown",
    )
    return ONBOARDING_WEIGHT


async def onboarding_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        weight: float | None = float(update.message.text.strip())
    except (ValueError, AttributeError):
        weight = None
    context.chat_data["ob_weight"] = weight

    keyboard = [
        [InlineKeyboardButton("Muscle Gain / Hypertrophy", callback_data="muscle_gain")],
        [InlineKeyboardButton("Fat Loss / Body Recomposition", callback_data="fat_loss")],
        [InlineKeyboardButton("Endurance / Marathon Prep", callback_data="marathon_prep")],
        [InlineKeyboardButton("General Fitness / Maintenance", callback_data="maintenance")],
    ]
    await update.message.reply_text(
        "What is your primary fitness goal?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return ONBOARDING_GOAL


async def onboarding_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.chat_data["ob_goal"] = query.data

    keyboard = [
        [InlineKeyboardButton("Beginner (under 1 year)", callback_data="beginner")],
        [InlineKeyboardButton("Intermediate (1–3 years)", callback_data="intermediate")],
        [InlineKeyboardButton("Advanced (3+ years)", callback_data="advanced")],
    ]
    await query.edit_message_text(
        "What is your training experience level?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return ONBOARDING_LEVEL


async def onboarding_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.chat_data["ob_level"] = query.data
    await query.edit_message_text(
        "Do you have any injuries, medical conditions, or physical limitations "
        "I should be aware of?\n(Reply none if nothing applies)",
    )
    return ONBOARDING_MEDICAL


async def onboarding_medical(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw = update.message.text.strip().lower()
    medical: str | None = (
        None if raw in ("", "none", "no", "n/a", "na") else update.message.text.strip()
    )

    cd = context.chat_data
    _save_profile(
        name=cd["ob_name"],
        dob=cd["ob_dob"],
        weight=cd.get("ob_weight"),
        goal=cd["ob_goal"],
        fitness_level=cd["ob_level"],
        medical=medical,
    )

    athlete_ctx = get_athlete_context()
    context.chat_data["athlete_ctx"] = athlete_ctx

    await update.message.reply_text(
        f"Profile saved! Welcome, *{cd['ob_name']}*.\n\n"
        "Your coaching session is ready. Ask me anything about your training, "
        "nutrition, recovery, or wellbeing.",
        parse_mode="Markdown",
    )
    return CHAT


# ── Chat handler ──────────────────────────────────────────────────────────────

async def chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _stream_reply(update, context, update.message.text)
    return CHAT


# ── Fallback / cancel ─────────────────────────────────────────────────────────

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Session cancelled. Send /start to begin again.")
    return ConversationHandler.END


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Send /start to begin your coaching session.")


# ── Health server (port 8000, background thread) ──────────────────────────────

def _start_health_server() -> None:
    health_app = FastAPI(title="FlexLLM health", docs_url=None, redoc_url=None)
    health_app.include_router(health_routes.router)
    uvicorn.run(health_app, host="0.0.0.0", port=8000, log_level="warning")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    setup_tracing()

    threading.Thread(target=_start_health_server, daemon=True, name="health-server").start()
    logger.info("Health server starting on port 8000")

    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            ONBOARDING_NAME:    [MessageHandler(filters.TEXT & ~filters.COMMAND, onboarding_name)],
            ONBOARDING_DOB:     [MessageHandler(filters.TEXT & ~filters.COMMAND, onboarding_dob)],
            ONBOARDING_WEIGHT:  [MessageHandler(filters.TEXT & ~filters.COMMAND, onboarding_weight)],
            ONBOARDING_GOAL:    [CallbackQueryHandler(onboarding_goal)],
            ONBOARDING_LEVEL:   [CallbackQueryHandler(onboarding_level)],
            ONBOARDING_MEDICAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboarding_medical)],
            CHAT:               [MessageHandler(filters.TEXT & ~filters.COMMAND, chat_message)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_chat=True,
    )

    application.add_handler(conv)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_message)
    )

    logger.info("FlexLLM Telegram bot starting (polling)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
