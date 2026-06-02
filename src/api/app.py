"""FastAPI sub-application — health and readiness endpoints only.

The primary entry point is chainlit_app.py (Chainlit server).
This module is imported by chainlit_app.py to register /health routes
on Chainlit's internal FastAPI instance.
"""
import logging
import logging.config

from src.api.routes import health

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
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "uvicorn.access": {"level": "WARNING"},
    },
})


def register_health_routes(fastapi_app) -> None:
    """Attach /health and /health/ready to any FastAPI app instance."""
    fastapi_app.include_router(health.router)
