import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    phone: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_active: datetime = field(default_factory=datetime.utcnow)
    # Message count at the moment this session window began (for summary slicing).
    # None until lazily initialized from the graph checkpointer on first message.
    initial_message_count: int | None = None


class SessionManager:
    """Per-conversation state: asyncio locking, activity tracking, inactivity detection.

    One instance lives for the lifetime of the FastAPI process. Each unique WhatsApp
    phone number gets its own ConversationSession with an exclusive asyncio.Lock so
    messages are processed strictly in-order per conversation.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._mu = asyncio.Lock()

    async def get_or_create(self, phone: str) -> ConversationSession:
        async with self._mu:
            if phone not in self._sessions:
                self._sessions[phone] = ConversationSession(phone=phone)
                logger.info("New conversation session", extra={"phone": phone})
            return self._sessions[phone]

    def touch(self, phone: str) -> None:
        if phone in self._sessions:
            self._sessions[phone].last_active = datetime.utcnow()

    def get_inactive(self, timeout_minutes: int) -> list[ConversationSession]:
        cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        return [s for s in self._sessions.values() if s.last_active < cutoff]

    async def remove(self, phone: str) -> None:
        async with self._mu:
            self._sessions.pop(phone, None)
            logger.info("Session removed", extra={"phone": phone})

    @property
    def active_count(self) -> int:
        return len(self._sessions)
