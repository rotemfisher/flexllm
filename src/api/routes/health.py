import asyncio
import sqlite3

from fastapi import APIRouter

from src.api import dependencies
from src.config import config

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def liveness():
    """Returns 200 whenever the process is alive."""
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness():
    """Checks all downstream dependencies. Returns 200 only when fully operational."""
    checks: dict[str, str] = {}
    overall_ok = True

    # ── SQLite database ───────────────────────────────────────────────────────
    try:
        def _ping_db():
            con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
            con.execute("SELECT 1")
            con.close()

        await asyncio.to_thread(_ping_db)
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        overall_ok = False

    # ── LangGraph compiled graph ──────────────────────────────────────────────
    if dependencies.compiled_graph is not None:
        checks["graph"] = "ok"
    else:
        checks["graph"] = "not initialized"
        overall_ok = False

    # ── WhatsApp bridge ───────────────────────────────────────────────────────
    try:
        client = dependencies.get_whatsapp_client()
        status = await client.get_status()
        checks["whatsapp"] = status.get("state", "unknown")
    except Exception as exc:
        checks["whatsapp"] = f"error: {exc}"
        # Degraded but not fatal — still can serve if bridge is temporarily down
        overall_ok = False

    return {
        "status": "ok" if overall_ok else "degraded",
        "checks": checks,
        "active_sessions": dependencies.session_manager.active_count if dependencies.session_manager else 0,
    }
