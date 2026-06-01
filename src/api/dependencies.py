"""App-level singletons injected via FastAPI Depends().

All attributes are populated during the lifespan startup hook in app.py and
are None before that point. Importing this module at startup is safe.
"""
from __future__ import annotations

from src.api.services.session_manager import SessionManager
from src.api.services.whatsapp_client import WhatsAppClient

compiled_graph = None
whatsapp_client: WhatsAppClient | None = None
session_manager: SessionManager | None = None


def get_graph():
    return compiled_graph


def get_whatsapp_client() -> WhatsAppClient:
    assert whatsapp_client is not None, "WhatsApp client not initialized"
    return whatsapp_client


def get_session_manager() -> SessionManager:
    assert session_manager is not None, "Session manager not initialized"
    return session_manager
