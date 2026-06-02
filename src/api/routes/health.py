import asyncio
import sqlite3

from fastapi import APIRouter

from src.api import dependencies
from src.config import config

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def liveness():
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness():
    checks: dict[str, str] = {}
    overall_ok = True

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

    if dependencies.compiled_graph is not None:
        checks["graph"] = "ok"
    else:
        checks["graph"] = "not initialized"
        overall_ok = False

    return {
        "status": "ok" if overall_ok else "degraded",
        "checks": checks,
    }
