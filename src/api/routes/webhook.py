import asyncio
import functools
import logging
import re
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, status
from langchain_core.messages import AIMessage

from src.agent.coach_agent import build_coach_graph, get_athlete_context
from src.agent.memory import SummaryStore, save_session_summary, maybe_refresh_weekly_summary
from src.api.dependencies import get_graph, get_session_manager, get_whatsapp_client
from src.api.models.webhook import IncomingWhatsAppMessage, WebhookResponse
from src.api.services.session_manager import SessionManager
from src.api.services.whatsapp_client import WhatsAppClient
from src.config import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhook"])

_AGENT_NODES = frozenset({"trainer", "physiotherapist", "recovery_coach", "dietitian"})

_NON_TEXT_REPLY = (
    "I can only read text messages right now. "
    "Please type your question and I'll be happy to help!"
)
_ERROR_REPLY = (
    "Something went wrong on my end. Please try again in a moment."
)
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


# ── Auth ──────────────────────────────────────────────────────────────────────

def _verify_secret(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {config.WEBHOOK_SECRET}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing webhook secret",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_phone(raw: str) -> str:
    """Strip WhatsApp suffixes and leading + for consistent use as thread_id."""
    return raw.replace("@c.us", "").replace("@s.whatsapp.net", "").lstrip("+")


def _is_allowed(phone: str) -> bool:
    allowed = config.allowed_numbers_set
    return not allowed or phone in allowed


def _extract_response(result: dict) -> tuple[str, str | None]:
    """Return (response_text, active_agent) from a graph.invoke() result."""
    messages = result.get("messages", [])
    active_agent = result.get("active_agent")
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return content.strip(), active_agent
    return _ERROR_REPLY, active_agent


def _format_for_whatsapp(text: str, agent: str | None) -> str:
    """Adapt LLM markdown output for WhatsApp's limited renderer."""
    # Strip markdown headers (## Heading → plain text)
    text = _MARKDOWN_HEADER_RE.sub("", text)
    # Prefix with agent label in WhatsApp bold
    if agent:
        label = agent.replace("_", " ").title()
        text = f"*{label}*\n{text}"
    return text.strip()


def _persist_summary(graph, run_config: dict, initial_count: int) -> None:
    """Summarize session messages — called in a thread pool, never raises."""
    try:
        state = graph.get_state(run_config)
        all_msgs = list(state.values.get("messages", []))
        new_msgs = all_msgs[initial_count:]
        if len(new_msgs) < 4:
            return
        active_agent = state.values.get("active_agent") or "trainer"
        store = SummaryStore(config.DB_PATH)
        save_session_summary(new_msgs, active_agent, store, config.MODEL_ID)
        maybe_refresh_weekly_summary(store, config.MODEL_ID)
        logger.info("Session summary saved", extra={"agent": active_agent, "messages": len(new_msgs)})
    except Exception:
        logger.exception("Session summary generation failed — skipping")


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/webhook/whatsapp", response_model=WebhookResponse)
async def whatsapp_webhook(
    payload: IncomingWhatsAppMessage,
    graph=Depends(get_graph),
    wa: WhatsAppClient = Depends(get_whatsapp_client),
    sessions: SessionManager = Depends(get_session_manager),
    _auth: None = Depends(_verify_secret),
) -> WebhookResponse:
    request_id = uuid.uuid4().hex[:8]
    raw_from = payload.from_number
    phone = _normalize_phone(raw_from)

    logger.info(
        "Incoming message",
        extra={"request_id": request_id, "phone": phone, "type": payload.type},
    )

    # ── Access control ────────────────────────────────────────────────────────
    if not _is_allowed(phone):
        logger.warning("Rejected unlisted number", extra={"phone": phone})
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Number not allowed")

    # ── Non-text messages ─────────────────────────────────────────────────────
    if payload.type != "chat" or not payload.body.strip():
        await wa.send_message(raw_from, _NON_TEXT_REPLY)
        return WebhookResponse(status="non_text_ignored", request_id=request_id)

    session = await sessions.get_or_create(phone)

    # Per-conversation lock ensures strictly sequential processing.
    async with session.lock:
        sessions.touch(phone)
        asyncio.create_task(wa.send_typing(raw_from))

        run_config = {
            "configurable": {"thread_id": phone},
            "run_name": f"coaching-whatsapp-{phone}",
            "tags": [config.ENVIRONMENT, "whatsapp", config.MODEL_ID],
            "metadata": {
                "source": "whatsapp",
                "phone": phone,
                "request_id": request_id,
                "model": config.MODEL_ID,
                "environment": config.ENVIRONMENT,
            },
        }

        # Lazily capture the message count at session start so summaries only
        # cover messages exchanged in this session window.
        if session.initial_message_count is None:
            try:
                def _get_count():
                    s = graph.get_state(run_config)
                    return len(list(s.values.get("messages", [])))
                session.initial_message_count = await asyncio.to_thread(_get_count)
            except Exception:
                session.initial_message_count = 0

        # Refresh athlete context on every turn (profile may have been updated).
        athlete_ctx = await asyncio.to_thread(get_athlete_context)

        try:
            invoke_fn = functools.partial(
                graph.invoke,
                {"messages": [("human", payload.body)], "athlete_context": athlete_ctx},
                config=run_config,
            )
            result = await asyncio.to_thread(invoke_fn)
        except Exception:
            logger.exception(
                "Graph invocation failed",
                extra={"request_id": request_id, "phone": phone},
            )
            await wa.send_message(raw_from, _ERROR_REPLY)
            return WebhookResponse(status="error", request_id=request_id)

        response_text, active_agent = _extract_response(result)
        formatted = _format_for_whatsapp(response_text, active_agent)

        await wa.send_message(raw_from, formatted)
        logger.info(
            "Reply sent",
            extra={
                "request_id": request_id,
                "phone": phone,
                "agent": active_agent,
                "length": len(formatted),
            },
        )
        return WebhookResponse(status="ok", request_id=request_id)
