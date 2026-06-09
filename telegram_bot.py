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
- Parse modes: static bot messages use HTML. LLM streaming output is shown as
  plain text (stripped of markdown symbols) during streaming, then converted to
  Telegram HTML for the final edit. Long responses are split across multiple
  messages at paragraph boundaries to stay within Telegram's 4096-char limit.
- Health server: a minimal FastAPI app runs on port 8000 in a daemon thread so
  Docker's healthcheck and watcher's depends_on still work.
"""
import asyncio
import datetime as dt
import html
import logging
import re
import threading
from zoneinfo import ZoneInfo

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

import psycopg
from psycopg.rows import dict_row

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
    ONBOARDING_HEIGHT,
    ONBOARDING_SEX,
    ONBOARDING_WEIGHT,
    ONBOARDING_GOAL,
    ONBOARDING_LEVEL,
    ONBOARDING_DIET,
    ONBOARDING_MEDICAL,
    CHAT,
) = range(10)

_AGENT_NODES = frozenset(
    {"trainer", "physiotherapist", "recovery_coach", "dietitian", "psychologist"}
)

# Handler-level filter — PTB drops messages from any other user before they
# reach any handler, so no LangGraph or DB work is ever triggered.
OWNER_FILTER = filters.User(user_id=config.TELEGRAM_ALLOWED_USER_ID)

# ── Schema migration (sync — runs before the event loop starts) ───────────────
# Each statement is guarded so adding an existing column is a no-op.
_MIGRATION_STMTS = [
    "ALTER TABLE athlete_profile ADD COLUMN IF NOT EXISTS name TEXT",
    "ALTER TABLE athlete_profile ADD COLUMN IF NOT EXISTS current_weight_kg REAL",
    "ALTER TABLE athlete_profile ADD COLUMN IF NOT EXISTS medical_conditions TEXT",
    "ALTER TABLE athlete_profile ADD COLUMN IF NOT EXISTS height_cm REAL",
    "ALTER TABLE athlete_profile ADD COLUMN IF NOT EXISTS biological_sex TEXT",
    "ALTER TABLE athlete_profile ADD COLUMN IF NOT EXISTS dietary_pref TEXT",
]
try:
    with psycopg.connect(config.DATABASE_URL) as _mig_con:
        for _stmt in _MIGRATION_STMTS:
            _mig_con.execute(_stmt)
except Exception as _exc:
    logger.warning("Schema migration: %s", _exc)


# ── DB helpers — sync functions, always call via asyncio.to_thread ────────────

def _onboarding_complete_sync() -> bool:
    try:
        with psycopg.connect(config.DATABASE_URL, row_factory=dict_row, autocommit=True) as con:
            row = con.execute(
                "SELECT onboarding_complete FROM athlete_profile ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return bool(row and row["onboarding_complete"] == 1)
    except Exception:
        return False


def _save_profile_sync(*, name, dob, height, sex, weight, goal, fitness_level, diet, medical) -> None:
    with psycopg.connect(config.DATABASE_URL) as con:
        con.execute(
            """
            INSERT INTO athlete_profile
                (name, date_of_birth, height_cm, biological_sex, current_goal, fitness_level,
                 current_weight_kg, dietary_pref, medical_conditions, onboarding_complete,
                 updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, NOW())
            """,
            (name, dob, height, sex, goal, fitness_level, weight, diet, medical),
        )


# ── Telegram formatting helpers ───────────────────────────────────────────────

_TG_MAX_LEN = 3900  # Telegram hard limit is 4096; 3900 gives headroom for HTML tags


def _quick_strip_md(text: str) -> str:
    """Strip the most disruptive Markdown symbols for plain-text intermediate display.

    Used during streaming so the user sees readable text instead of raw symbols
    while the full response is still being generated.
    """
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)  # ### headers
    text = re.sub(r'\*{1,3}|_{1,2}', '', text)                  # * ** *** _ __
    text = re.sub(r'^\s*\|[-:\s|]+\|\s*$', '', text, flags=re.MULTILINE)  # table separators
    text = re.sub(r'\|', '  ', text)                             # table pipes
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)  # horizontal rules
    text = re.sub(r'\\\[.+?\\\]', '', text, flags=re.DOTALL)    # LaTeX blocks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _md_to_tg_html(text: str) -> str:
    """Convert standard Markdown to Telegram-safe HTML.

    Telegram HTML supports <b>, <i>, <code>, <pre>. Everything else
    (headers, tables, LaTeX, horizontal rules) is converted to plain equivalents.
    """
    # 1. Protect fenced code blocks from HTML escaping
    code_blocks: dict[str, str] = {}

    def _save_fence(m: re.Match) -> str:
        key = f"\x00F{len(code_blocks)}\x00"
        code_blocks[key] = f"<pre>{html.escape(m.group(1))}</pre>"
        return key

    text = re.sub(r'```[^\n]*\n(.*?)```', _save_fence, text, flags=re.DOTALL)

    # 2. Protect inline code
    inline_codes: dict[str, str] = {}

    def _save_inline(m: re.Match) -> str:
        key = f"\x00I{len(inline_codes)}\x00"
        inline_codes[key] = f"<code>{html.escape(m.group(1))}</code>"
        return key

    text = re.sub(r'`([^`\n]+)`', _save_inline, text)

    # 3. HTML-escape everything outside the protected placeholders
    parts = re.split(r'(\x00[FI]\d+\x00)', text)
    text = ''.join(
        p if re.fullmatch(r'\x00[FI]\d+\x00', p)
        else p.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        for p in parts
    )

    # 4. Headers → bold (### Heading → <b>Heading</b>)
    # Strip any **...** wrapping already present inside the heading to avoid <b><b>...</b></b>.
    def _header_to_bold(m: re.Match) -> str:
        inner = m.group(1).strip()
        inner = re.sub(r'\*\*([^\n]+?)\*\*', r'\1', inner)
        return f"<b>{inner}</b>"

    text = re.sub(r'^#{1,6}\s*(.+)$', _header_to_bold, text, flags=re.MULTILINE)

    # 5. Tables: drop separator rows, convert pipes to spaces
    text = re.sub(r'^\s*\|[-:\s|]+\|\s*$', '', text, flags=re.MULTILINE)

    def _table_row(m: re.Match) -> str:
        cells = [c.strip() for c in m.group(0).strip('| \n').split('|')]
        return '  '.join(c for c in cells if c)

    text = re.sub(r'^\|.+$', _table_row, text, flags=re.MULTILINE)

    # 6. Bold: **text** or __text__
    text = re.sub(r'\*\*([^\n]+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^\n]+?)__', r'<b>\1</b>', text)

    # 7. Italic: *text* or _text_ (not crossing newlines)
    text = re.sub(r'(?<!\w)\*([^\n*]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^\n_]+?)_(?!\w)', r'<i>\1</i>', text)

    # 8. LaTeX math → plain text (strip delimiters, keep content)
    text = re.sub(r'\\\[(.+?)\\\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r'\1', text)

    # 9. Horizontal rules → blank line
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 10. Bullet list markers → •
    text = re.sub(r'^[\-\*]\s+', '• ', text, flags=re.MULTILINE)

    # 11. Restore protected code blocks
    for k, v in {**code_blocks, **inline_codes}.items():
        text = text.replace(k, v)

    # 12. Collapse extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _split_for_telegram(text: str, max_len: int = _TG_MAX_LEN) -> list[str]:
    """Split text into chunks ≤ max_len chars, breaking at paragraph boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current = ""

    for para in re.split(r'\n{2,}', text):
        if not para:
            continue
        sep = "\n\n" if current else ""
        candidate = current + sep + para
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) > max_len:
                for line in para.split('\n'):
                    sep = "\n" if current else ""
                    candidate = current + sep + line
                    if len(candidate) <= max_len:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = line[:max_len]
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks or [text[:max_len]]


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
    """Stream the coach reply, sending each specialist's response as its own message.

    Each agent (trainer, dietitian, …) gets a dedicated Telegram placeholder that
    is updated during streaming and finalised as a properly-formatted HTML message
    when that agent finishes.  Long single-agent responses are split at paragraph
    boundaries to stay within Telegram's 4096-char limit.
    """
    chat_id = update.effective_chat.id
    athlete_ctx = context.chat_data.get("athlete_ctx", "")
    run_cfg = _run_config(chat_id)
    graph = context.application.bot_data["graph"]

    # The very first placeholder is reused by the first agent that responds.
    placeholder = await update.effective_chat.send_message("⏳ Thinking…")

    agent_buf = ""
    current_agent: str | None = None
    current_label: str | None = None
    last_edit_at = 0.0
    MIN_EDIT_INTERVAL = 2.5

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _send(text: str, *, edit_msg=None) -> None:
        """Send or edit as HTML; fall back to stripped plain text on error."""
        try:
            if edit_msg:
                await edit_msg.edit_text(text, parse_mode="HTML")
            else:
                await update.effective_chat.send_message(text, parse_mode="HTML")
        except tg_error.BadRequest as e:
            if "Message is not modified" not in str(e):
                logger.warning("HTML send failed, retrying plain: %s", e)
            plain = _quick_strip_md(text)[:_TG_MAX_LEN]
            try:
                if edit_msg:
                    await edit_msg.edit_text(plain)
                else:
                    await update.effective_chat.send_message(plain)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Send failed: %s", e)

    async def _flush() -> None:
        """Convert current agent_buf to HTML and finalise the placeholder message.

        If the converted text exceeds _TG_MAX_LEN, extra paragraphs are sent as
        follow-up messages.  Resets `placeholder` to None when done.
        """
        nonlocal placeholder
        if not agent_buf.strip():
            try:
                await placeholder.delete()
            except Exception:
                pass
            placeholder = None
            return
        header = f"**[{current_label}]**\n" if current_label else ""
        html_chunks = _split_for_telegram(_md_to_tg_html(header + agent_buf))
        await _send(html_chunks[0], edit_msg=placeholder)
        for extra in html_chunks[1:]:
            await _send(extra)
        placeholder = None

    # ── stream ────────────────────────────────────────────────────────────────

    try:
        async for chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user_text)], "athlete_context": athlete_ctx},
            config=run_cfg,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")
            if node in _AGENT_NODES and isinstance(chunk, AIMessageChunk):
                if node != current_agent:
                    # Finalise previous agent's message (if any)
                    if current_agent is not None:
                        await _flush()
                        agent_buf = ""
                        # Fresh placeholder for the incoming specialist
                        placeholder = await update.effective_chat.send_message("⏳ Thinking…")

                    current_agent = node
                    current_label = node.replace("_", " ").title()

                if isinstance(chunk.content, str) and chunk.content:
                    agent_buf += chunk.content

                now = asyncio.get_running_loop().time()
                if agent_buf and placeholder and now - last_edit_at >= MIN_EDIT_INTERVAL:
                    preview = _quick_strip_md(f"**[{current_label}]**\n{agent_buf}")
                    try:
                        await placeholder.edit_text(preview[:_TG_MAX_LEN] + " ▌")
                        last_edit_at = now
                    except tg_error.BadRequest as e:
                        if "Message is not modified" not in str(e):
                            logger.warning("Streaming edit failed: %s", e)
                    except tg_error.RetryAfter as e:
                        logger.warning("Rate limited — backing off %ss", e.retry_after)
                        await asyncio.sleep(e.retry_after)
                    except Exception as e:
                        logger.warning("Unexpected streaming error: %s", e)

    except (GeneratorExit, asyncio.CancelledError):
        if placeholder:
            try:
                await placeholder.delete()
            except Exception:
                pass
        raise
    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        try:
            await (placeholder.edit_text if placeholder else update.effective_chat.send_message)(
                "Sorry, something went wrong. Please try again."
            )
        except Exception:
            pass
        return

    # Flush the last (or only) agent
    if current_agent:
        await _flush()
    elif placeholder:
        await placeholder.edit_text("(No response from coach)")


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
        "What is your height in centimetres?\n<i>(e.g. 175)</i>",
        parse_mode="HTML",
    )
    return ONBOARDING_HEIGHT


async def onboarding_height(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        height: float | None = float(update.message.text.strip())
    except (ValueError, AttributeError):
        height = None
    context.chat_data["ob_height"] = height

    keyboard = [
        [InlineKeyboardButton("Male", callback_data="male")],
        [InlineKeyboardButton("Female", callback_data="female")],
        [InlineKeyboardButton("Other / prefer not to say", callback_data="other")],
    ]
    await update.message.reply_text(
        "What is your biological sex? <i>(used only for calorie calculations)</i>",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
    )
    return ONBOARDING_SEX


async def onboarding_sex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.chat_data["ob_sex"] = query.data
    await query.edit_message_text(
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

    keyboard = [
        [InlineKeyboardButton("Omnivore (no restrictions)", callback_data="omnivore")],
        [InlineKeyboardButton("Vegetarian", callback_data="vegetarian")],
        [InlineKeyboardButton("Vegan", callback_data="vegan")],
        [InlineKeyboardButton("Gluten-free", callback_data="gluten-free")],
    ]
    await query.edit_message_text(
        "Do you follow any dietary preferences or restrictions?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return ONBOARDING_DIET


async def onboarding_diet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.chat_data["ob_diet"] = query.data
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
        height=cd.get("ob_height"),
        sex=cd.get("ob_sex"),
        weight=cd.get("ob_weight"),
        goal=cd["ob_goal"],
        fitness_level=cd["ob_level"],
        diet=cd.get("ob_diet"),
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


# ── Proactive scheduler ───────────────────────────────────────────────────────

def _setup_scheduler(application: Application) -> None:
    """Register all proactive coaching jobs with PTB's built-in JobQueue."""
    from src.scheduler.jobs import (
        morning_briefing_job,
        preworkout_reminder_job,
        daily_nutrition_job,
        evening_summary_job,
        weekend_review_job,
    )

    tz = ZoneInfo(config.SCHEDULER_TIMEZONE)
    jq = application.job_queue

    # Morning briefing: 07:00 — readiness + workout recommendation
    jq.run_daily(morning_briefing_job,    time=dt.time(7,  0, tzinfo=tz))
    # Daily nutrition plan: 07:30
    jq.run_daily(daily_nutrition_job,     time=dt.time(7, 30, tzinfo=tz))
    # Pre-workout briefing: 17:30 (30 min before default 18:00 workout slot)
    jq.run_daily(preworkout_reminder_job, time=dt.time(17, 30, tzinfo=tz))
    # Evening summary: 21:00
    jq.run_daily(evening_summary_job,     time=dt.time(21,  0, tzinfo=tz))
    # Weekly review: Saturdays at 22:00 only (days tuple: 0=Sun … 6=Sat in PTB)
    jq.run_daily(weekend_review_job,      time=dt.time(22,  0, tzinfo=tz), days=(6,))

    logger.info("Proactive coaching scheduler initialised (tz=%s)", config.SCHEDULER_TIMEZONE)


# ── Weekly plan approval callbacks ────────────────────────────────────────────

async def weekly_plan_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline-button responses to the Saturday-night weekly plan message."""
    from src.scheduler.weekly_supervisor import WEEKLY_PLAN_APPROVED, WEEKLY_PLAN_ADJUST
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    if query.data == WEEKLY_PLAN_APPROVED:
        await update.effective_chat.send_message(
            "✅ Plan confirmed — have a strong week! 💪",
            parse_mode="HTML",
        )
    elif query.data == WEEKLY_PLAN_ADJUST:
        await update.effective_chat.send_message(
            "Got it. Send /start and tell me which sessions you'd like to change.",
            parse_mode="HTML",
        )


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
    _setup_scheduler(application)


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
            ONBOARDING_HEIGHT:  [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_height)],
            ONBOARDING_SEX:     [CallbackQueryHandler(onboarding_sex, pattern="^(male|female|other)$")],
            ONBOARDING_WEIGHT:  [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_weight)],
            ONBOARDING_GOAL:    [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_goal)],
            ONBOARDING_LEVEL:   [CallbackQueryHandler(onboarding_level, pattern="^(beginner|intermediate|advanced)$")],
            ONBOARDING_DIET:    [CallbackQueryHandler(onboarding_diet, pattern="^(omnivore|vegetarian|vegan|gluten-free)$")],
            ONBOARDING_MEDICAL: [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, onboarding_medical)],
            CHAT:               [MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, chat_message)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel, filters=OWNER_FILTER)],
        per_chat=True,
        per_message=False,
    )

    application.add_handler(conv)
    # Weekly plan approval buttons — registered before the catch-all so they
    # are not swallowed by the ConversationHandler's fallback paths.
    application.add_handler(
        CallbackQueryHandler(weekly_plan_callback, pattern="^weekly_plan:")
    )
    # Catch-all for anyone who messages before /start (or unknown users — they
    # won't pass OWNER_FILTER in the ConversationHandler so they end up here).
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & OWNER_FILTER, unknown_message)
    )

    logger.info("FlexLLM Telegram bot starting (polling)…")
    application.run_polling(allowed_updates=Update.ALL_TYPES, bootstrap_retries=-1, drop_pending_updates=True)


if __name__ == "__main__":
    main()
