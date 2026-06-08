import asyncio
from pathlib import Path

import psycopg

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.api import dependencies
from src.config import config

router = APIRouter(tags=["health"])

# Written by the cloudflared sidecar when the tunnel URL is known.
_TUNNEL_URL_FILE = Path("/data/tunnel_url.txt")


@router.get("/health", summary="Liveness probe")
async def liveness():
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness():
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        def _ping_db():
            with psycopg.connect(config.DATABASE_URL, autocommit=True) as con:
                con.execute("SELECT 1")

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


@router.get("/remote-url", response_class=HTMLResponse, summary="Current tunnel URL",
            include_in_schema=False)
async def remote_url():
    """
    Human-readable page showing the current Cloudflare tunnel URL.
    Open http://localhost:8000/remote-url on your Mac to get the link
    you can open on your phone over cellular.
    """
    try:
        content = _TUNNEL_URL_FILE.read_text().strip()
    except FileNotFoundError:
        content = "pending"

    if content == "pending":
        body = "<p>⏳ Tunnel is connecting… refresh in a few seconds.</p>"
        url_display = ""
    elif content.startswith("https://"):
        body = f"""
            <p>Open this URL on your phone (works on WiFi and cellular):</p>
            <p style="font-size:1.4em; word-break:break-all;">
                <a href="{content}">{content}</a>
            </p>
            <p style="color:#888; font-size:0.85em;">
                This URL changes every time Docker restarts. Bookmark this page
                (<code>http://YOUR-MAC-IP:8000/remote-url</code>) to always find the latest link.
            </p>
        """
        url_display = content
    else:
        body = f"<p>⚠️ Tunnel error: {content}</p>"
        url_display = ""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FlexLLM — Remote URL</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; }}
    a {{ color: #2563eb; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
  </style>
  {'<meta http-equiv="refresh" content="5">' if not url_display else ''}
</head>
<body>
  <h2>📱 FlexLLM Remote Access</h2>
  {body}
</body>
</html>"""
