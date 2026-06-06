"""FlexLLM Coach — Telegram bot entry point (polling mode).

Run with:
    python telegram_bot.py

Architecture notes:
- Owner-only filter: TELEGRAM_ALLOWED_USER_ID in .env — all other senders are
  silently ignored at the Handler level, wasting zero LangGraph resources.
- Async-safe DB: every sqlite3 call runs via asyncio.to_thread so the event
  loop is never blocked.
- Streaming: agent chunks are buffered and edited into a placeholder message
  every MIN_EDIT_INTERVAL seconds. Telegram-specific errors (MessageNotModified,
  RetryAfter) are handled explicitly; other errors are logged rather than
  swallowed silently.
- Parse modes: static bot messages use HTML (safe to include user-supplied text
  via html.escape). LLM-generated streaming output uses Markdown with a plain-
  text fallback because LLM output may contain unmatched Markdown tokens.
- Health server: a minimal FastAPI app runs on port 8000 in a daemon thread so
  Docker's healthcheck and watcher's depends_on still work.
"""
import asyncio
import html
import logging
import sqlite3
import threading

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import AIMessageChunk, HumanMessage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram import error as tg_error
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

# Handler-level filter — PTB drops messages from any other user before they
# reach any handler, so no LangGraph or DB work is ever triggered.
OWNER_FILTER = filters.User(user_id=config.TELEGRAM_ALLOWED_USER_ID)

# ── Schema migration (sync — runs before the event loop starts) ───────────────
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


# ── DB helpers — sync functions, always call via asyncio.to_thread ────────────

def _onboarding_complete_sync() -> bool:
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        row = con.execute(
            "SELECT onboarding_complete FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
        return bool(row and row[0] == 1)
    except Exception:
        return False


def _save_profile_sync(*, name, dob, weight, goal, fitness_level, medical) -> None:
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
    """Stream the coach reply into a Telegram placeholder message.

    Intermediate edits use plain text to avoid broken-Markdown errors mid-
    stream. The final edit tries Markdown (LLM output style) with a plain-text
    fallback. Telegram-specific errors are handled individually so real bugs
    aren't silently swallowed.
    """
    chat_id = update.effective_chat.id
    athlete_ctx = context.chat_data.get("athlete_ctx", "")
    run_cfg = _run_config(chat_id)
    graph = context.application.bot_data["graph"]

    placeholder = await update.effective_chat.send_message("⏳ Thinking…")

    buffer = ""
    current_agent: str | None = None
    last_edit_at = 0.0
    MIN_EDIT_INTERVAL = 2.5  # seconds — well within Telegram's ~1 edit/s limit

    try:
        async for chunk, metadata in graph.astream(
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
                    except tg_error.BadRequest as e:
                        if "Message is not modified" not in str(e):
                            logger.warning("Streaming edit failed: %s", e)
                    except tg_error.RetryAfter as e:
                        logger.warning("Rate limited — backing off %ss", e.retry_after)
                        await asyncio.sleep(e.retry_after)
                    except Exception as e:
                        logger.warning("Unexpected streaming error: %s", e)
    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        await placeholder.edit_text("Sorry, something went wrong. Please try again.")
        return

    if not buffer:
        buffer = "(No response from coach)"

    # Final edit: LLM output is Markdown — try it, fall back to plain text.
    try:
        await placeholder.edit_text(buffer, parse_mode="Markdown")
    except tg_error.BadRequest as e:
        if "Message is not modified" not in str(e):
            logger.warning("Final Markdown edit failed, retrying plain: %s", e)
        try:
            await placeholder.edit_text(buffer)
        except Exception:
            pass
    except Exception as e:
        logger.warning("Final edit failed: %s", e)
        try:
            await placeholder.edit_text(buffer)
        except Exception:
            pass


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    already_done = await asyncio.to_thread(_onboarding_complete_sync)
    if already_done:
        athlete_ctx = await asyncio.to_thread(get_athlete_context)
        context.chat_data["athlete_ctx"] = athlete_ctx
        await update.message.reply_text(
            "<b>FlexLLM Coach is ready.</b>\n\n"
            "Ask me anything about your training, nutrition, recovery, or wellbeing.\n\n"
            "---\n"
            f"{html.escape(athlete_ctx)}",
            parse_mode="HTML",
        )
        return CHAT

    await update.message.reply_text(
        "<b>Welcome to FlexLLM Coach!</b>\n\n"
        "Before we start I need a few details to personalise your experience. "
        "This only happens once and takes about 2 minutes.\n\n"
        "What is your name?",
        parse_mode="HTML",
    )
    return ONBOARDING_NAME


# ── Onboarding steps ──────────────────────────────────────────────────────────

async def onboarding_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    name = update.message.text.strip() or "Athlete"
    context.chat_data["ob_name"] = name
    await update.message.reply_text(
        f"Hi <b>{html.escape(name)}</b>! What is your date of birth?\n"
        "<i>(YYYY-MM-DD, e.g. 1995-03-21)</i>",
        parse_mode="HTML",
    )
    return ONBOARDING_DOB


async def onboarding_dob(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.chat_data["ob_dob"] = update.message.text.strip() or "1990-01-01"
    await update.message.reply_text(
        "What is your current weight in kilograms?\n<i>(e.g. 78.5)</i>",
        parse_mode="HTML",
    )
    return ONBOARDING_WEIGHT


async def onboarding_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        weight: float | None = float(update.message.text.strip())
    except (ValueError, AttributeError):
        weight = None
    context.chat_data["ob_weight"] = weight

    await update.message.reply_text(
        "What is your primary fitness goal?\n\n"
        "<i>Tell me in your own words — e.g. \"lose 5 kg before summer\", "
        "\"run my first marathon\", \"build muscle and feel stronger\", etc.</i>",
        parse_mode="HTML",
    )
    return ONBOARDING_GOAL


async def onboarding_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.chat_data["ob_goal"] = update.message.text.strip() or "general fitness"

    keyboard = [
        [InlineKeyboardButton("Beginner (under 1 year)", callback_data="beginner")],
        [InlineKeyboardButton("Intermediate (1–3 years)", callback_data="intermediate")],
        [InlineKeyboardButton("Advanced (3+ years)", callback_data="advanced")],
    ]
    await update.message.reply_text(
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
        "I should be aware of?\n(Reply <i>none</i> if nothing applies)",
        parse_mode="HTML",
    )
    return ONBOARDING_MEDICAL


async def onboarding_medical(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw = update.message.text.strip().lower()
    medical: str | None = (
        None if raw in ("", "none", "no", "n/a", "na") else update.message.text.strip()
    )

    cd = context.chat_data
    await asyncio.to_thread(
        _save_profile_sync,
        name=cd["ob_name"],
        dob=cd["ob_dob"],
        weight=cd.get("ob_weight"),
        goal=cd["ob_goal"],
        fitness_level=cd["ob_level"],
        medical=medical,
    )

    athlete_ctx = await asyncio.to_thread(get_athlete_context)
    context.chat_data["athlete_ctx"] = athlete_ctx

    await update.message.reply_text(
        f"Profile saved! Welcome, <b>{html.escape(cd['ob_name'])}</b>.\n\n"
        "Your coaching session is ready. Ask me anything about your training, "
        "nutrition, recovery, or wellbeing.",
        parse_mode="HTML",
    )
    return CHAT


# ── Chat ──────────────────────────────────────────────────────────────────────

async def chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await _stream_reply(update, context, update.message.text)
    return CHAT


# ── Fallbacks ─────────────────────────────────────────────────────────────────

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Session cancelled. Send /start to begin again.")
    return ConversationHandler.END


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Send /start to begin your coaching session.")


# ── Health server (port 8000, daemon thread) ──────────────────────────────────

def _start_health_server() -> None:
    health_app = FastAPI(title="FlexLLM health", docs_url=None, redoc_url=None)
    health_app.include_router(health_routes.router)
    uvicorn.run(health_app, host="0.0.0.0", port=8000, log_level="warning")


# ── Entry point ───────────────────────────────────────────────────────────────

async def _post_init(application: Application) -> None:
    ctx = build_coach_graph()
    graph = await ctx.__aenter__()
    application.bot_data["graph"] = graph
    application.bot_data["_graph_ctx"] = ctx
    _api_pkg.dependencies.compiled_graph = graph


async def _post_shutdown(application: Application) -> None:
    ctx = application.bot_data.get("_graph_ctx")
    if ctx:
        await ctx.__aexit__(None, None, None)


def main() -> None:
    setup_logging()
    setup_tracing()

    threading.Thread(target=_start_health_server, daemon=True, name="health-server").start()
    logger.info("Health server starting on port 8000")

    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start, filters=OWNER_FILTER)],
        states={
            ONBOARDING_NAME:    [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_name)],
            ONBOARDING_DOB:     [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_dob)],
            ONBOARDING_WEIGHT:  [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_weight)],
            ONBOARDING_GOAL:    [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_goal)],
            ONBOARDING_LEVEL:   [CallbackQueryHandler(onboarding_level)],
            ONBOARDING_MEDICAL: [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_medical)],
            CHAT:               [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, chat_message)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel, filters=OWNER_FILTER)],
        per_chat=True,
        per_message=False,
    )

    application.add_handler(conv)
    # Catch-all for anyone who messages before /start (or unknown users — they
    # won't pass OWNER_FILTER in the ConversationHandler so they end up here).
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, unknown_message)
    )

    logger.info("FlexLLM Telegram bot starting (polling)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, bootstrap_retries=-1, drop_pending_updates=True)


if __name__ == "__main__":
    main()
