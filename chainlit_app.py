"""FlexLLM Coach — Chainlit entry point.

Run with:
    chainlit run chainlit_app.py --host 0.0.0.0 --port 8000

Startup order:
  1. build_coach_graph() opens SqliteSaver (kept alive via ExitStack for process lifetime).
  2. Schema migration adds new onboarding columns to athlete_profile if absent.
  3. /health + /health/ready are registered on Chainlit's internal FastAPI instance.
  4. Per-user: on_chat_start checks onboarding state, routes to form or chat.
"""
import atexit
import logging
import sqlite3
import uuid
from contextlib import ExitStack

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.app import register_health_routes
from src.agent.coach_agent import build_coach_graph, get_athlete_context
from src.config import config
from src.tracing import setup_tracing

logger = logging.getLogger(__name__)

_AGENT_NODES = frozenset({"trainer", "physiotherapist", "recovery_coach", "dietitian", "psychologist"})

# ── One-time process startup ───────────────────────────────────────────────────

setup_tracing()

# One graph instance shared across all users. SqliteSaver connection is kept
# open by ExitStack and closed on process exit.
_exit_stack = ExitStack()
_graph = _exit_stack.enter_context(build_coach_graph())
atexit.register(_exit_stack.close)

# Migrate schema: add columns introduced for the web onboarding flow.
# Try each statement individually; OperationalError means the column exists already.
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
            pass  # column already exists — fine
    _mig_con.commit()
    _mig_con.close()
except Exception as _exc:
    logger.warning("Schema migration: %s", _exc)

# Register /health routes and security middleware on Chainlit's Starlette app.
try:
    from chainlit.server import app as _cl_server  # type: ignore[import]

    register_health_routes(_cl_server)

    class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            return response

    _cl_server.add_middleware(_SecurityHeadersMiddleware)
except Exception as _exc:
    logger.warning("Server configuration: %s", _exc)


# ── Auth ──────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Single-password gate. Set APP_PASSWORD in .env."""
    if password == config.APP_PASSWORD:
        return cl.User(identifier="admin", metadata={"role": "admin"})
    return None


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


def _save_profile(
    *,
    name: str,
    dob: str,
    weight: float | None,
    goal: str,
    fitness_level: str,
    medical: str | None,
) -> None:
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


# ── Session initialisation ────────────────────────────────────────────────────

async def _start_coaching_session() -> None:
    # Use Chainlit's thread_id as the LangGraph checkpoint key so each
    # conversation in the sidebar has its own agent state.
    thread_id = getattr(cl.context.session, "thread_id", None) or str(uuid.uuid4())
    run_config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"coaching-{thread_id[:8]}",
        "tags": [config.ENVIRONMENT, "chainlit", config.MODEL_ID],
        "metadata": {"source": "chainlit", "thread_id": thread_id},
    }
    athlete_ctx = get_athlete_context()
    cl.user_session.set("run_config", run_config)
    cl.user_session.set("athlete_ctx", athlete_ctx)

    await cl.Message(
        content=(
            "**FlexLLM Coach is ready.**\n\n"
            "Ask me anything about your training, nutrition, recovery, or wellbeing.\n\n"
            "---\n"
            f"{athlete_ctx}"
        )
    ).send()


# ── Onboarding flow ────────────────────────────────────────────────────────────

async def _run_onboarding() -> None:
    await cl.Message(
        content=(
            "**Welcome to FlexLLM Coach!**\n\n"
            "Before we start I need a few details to personalise your experience. "
            "This only happens once and takes about 2 minutes."
        )
    ).send()

    # Step 1 — Name
    res = await cl.AskUserMessage(content="What is your name?", timeout=600).send()
    name = (res["output"].strip() if res else "") or "Athlete"

    # Step 2 — Date of birth
    res = await cl.AskUserMessage(
        content=f"Hi **{name}**! What is your date of birth? (YYYY-MM-DD, e.g. 1995-03-21)",
        timeout=600,
    ).send()
    dob = (res["output"].strip() if res else "") or "1990-01-01"

    # Step 3 — Current weight
    res = await cl.AskUserMessage(
        content="What is your current weight in kilograms? (e.g. 78.5)",
        timeout=600,
    ).send()
    try:
        weight: float | None = float(res["output"].strip()) if res else None
    except (ValueError, AttributeError):
        weight = None

    # Step 4 — Primary goal
    res = await cl.AskActionMessage(
        content="What is your primary fitness goal?",
        actions=[
            cl.Action(name="muscle_gain",  value="muscle_gain",  label="Muscle Gain / Hypertrophy"),
            cl.Action(name="fat_loss",     value="fat_loss",     label="Fat Loss / Body Recomposition"),
            cl.Action(name="marathon_prep",value="marathon_prep",label="Endurance / Marathon Prep"),
            cl.Action(name="maintenance",  value="maintenance",  label="General Fitness / Maintenance"),
        ],
        timeout=600,
    ).send()
    goal = (res.get("value") if res else None) or "maintenance"

    # Step 5 — Experience level
    res = await cl.AskActionMessage(
        content="What is your training experience level?",
        actions=[
            cl.Action(name="beginner",     value="beginner",     label="Beginner (under 1 year)"),
            cl.Action(name="intermediate", value="intermediate", label="Intermediate (1–3 years)"),
            cl.Action(name="advanced",     value="advanced",     label="Advanced (3+ years)"),
        ],
        timeout=600,
    ).send()
    fitness_level = (res.get("value") if res else None) or "intermediate"

    # Step 6 — Medical constraints (optional)
    res = await cl.AskUserMessage(
        content=(
            "Do you have any injuries, medical conditions, or physical limitations "
            "I should be aware of?\n(Type **none** if nothing applies)"
        ),
        timeout=600,
    ).send()
    medical_raw = (res["output"].strip() if res else "").lower()
    medical: str | None = None if medical_raw in ("", "none", "no", "n/a", "na") else medical_raw

    _save_profile(
        name=name,
        dob=dob,
        weight=weight,
        goal=goal,
        fitness_level=fitness_level,
        medical=medical,
    )

    await cl.Message(
        content=f"Profile saved! Welcome, **{name}**. Your coaching session is ready."
    ).send()

    await _start_coaching_session()


# ── Chat lifecycle ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    if _onboarding_complete():
        await _start_coaching_session()
    else:
        await _run_onboarding()


@cl.on_message
async def on_message(message: cl.Message):
    run_config = cl.user_session.get("run_config")
    athlete_ctx = cl.user_session.get("athlete_ctx")

    if not run_config:
        await cl.Message(content="Session not initialised — please refresh the page.").send()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    current_agent: str | None = None
    async for chunk, metadata in _graph.astream(
        {"messages": [HumanMessage(content=message.content)], "athlete_context": athlete_ctx},
        config=run_config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        if node in _AGENT_NODES and isinstance(chunk, AIMessageChunk):
            if node != current_agent:
                if current_agent is not None:
                    await response_msg.stream_token("\n\n")
                label = node.replace("_", " ").title()
                await response_msg.stream_token(f"**[{label}]**\n")
                current_agent = node
            if isinstance(chunk.content, str) and chunk.content:
                await response_msg.stream_token(chunk.content)

    await response_msg.update()
