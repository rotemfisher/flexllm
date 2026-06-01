"""FastAPI application factory for FlexLLM Coach — WhatsApp interface.

Lifecycle:
  startup  → compile LangGraph, init services, start inactivity cleanup loop
  running  → serve /webhook/whatsapp, /health, /health/ready
  shutdown → flush in-flight session summaries, close graph checkpointer
"""
import asyncio
import logging
import logging.config
from contextlib import ExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent.coach_agent import build_coach_graph
from src.api import dependencies
from src.api.routes import health, webhook
from src.api.services.session_manager import SessionManager
from src.api.services.whatsapp_client import WhatsAppClient
from src.config import config

# ── Logging ───────────────────────────────────────────────────────────────────

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    },
    "handlers": {
        "stdout": {"class": "logging.StreamHandler", "formatter": "default"}
    },
    "root": {"level": "INFO", "handlers": ["stdout"]},
    # Silence noisy third-party loggers
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "uvicorn.access": {"level": "WARNING"},
    },
})

logger = logging.getLogger(__name__)

# ── Background task: inactivity-triggered session summaries ───────────────────

async def _inactivity_cleanup(graph, session_mgr: SessionManager) -> None:
    """Every 10 minutes, summarize and evict sessions idle past the configured timeout."""
    while True:
        await asyncio.sleep(600)
        inactive = session_mgr.get_inactive(config.SESSION_TIMEOUT_MINUTES)
        for session in inactive:
            try:
                if session.initial_message_count is not None:
                    from src.api.routes.webhook import _persist_summary
                    run_config = {"configurable": {"thread_id": session.phone}}
                    await asyncio.to_thread(
                        _persist_summary, graph, run_config, session.initial_message_count
                    )
            except Exception:
                logger.exception("Cleanup summary failed for %s", session.phone)
            finally:
                await session_mgr.remove(session.phone)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FlexLLM API starting up (environment=%s)", config.ENVIRONMENT)

    with ExitStack() as stack:
        # build_coach_graph() is a sync context manager that opens SqliteSaver.
        graph = stack.enter_context(build_coach_graph())

        dependencies.compiled_graph = graph
        dependencies.whatsapp_client = WhatsAppClient()
        dependencies.session_manager = SessionManager()

        cleanup_task = asyncio.create_task(
            _inactivity_cleanup(graph, dependencies.session_manager)
        )
        logger.info("Startup complete — serving requests")

        try:
            yield
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    logger.info("FlexLLM API shut down cleanly")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="FlexLLM Coach API",
        version="1.0.0",
        description="Personal AI coaching assistant — WhatsApp interface",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Only the local WhatsApp bridge and localhost need cross-origin access.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3001", "http://whatsapp:3001"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(health.router)
    app.include_router(webhook.router)

    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
