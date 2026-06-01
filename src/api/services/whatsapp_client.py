import logging

import httpx

from src.config import config

logger = logging.getLogger(__name__)


class WhatsAppClient:
    """Async HTTP client for the whatsapp-web.js bridge service.

    All methods are fire-and-forget friendly: callers should catch exceptions
    if they need to degrade gracefully rather than propagate bridge errors.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or config.WHATSAPP_BRIDGE_URL).rstrip("/")

    async def send_message(self, to: str, message: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self._base_url}/send",
                json={"to": to, "message": message},
            )
            r.raise_for_status()
            logger.debug("Message sent", extra={"to": to, "length": len(message)})

    async def send_typing(self, to: str) -> None:
        """Signal typing state — best-effort, never raises."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{self._base_url}/typing", json={"to": to})
        except Exception:
            pass

    async def get_status(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self._base_url}/status")
            r.raise_for_status()
            return r.json()
