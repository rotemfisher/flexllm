"""
LangSmith tracing activation for FlexLLM.

LangChain/LangGraph read tracing settings from os.environ. Pydantic Settings
reads from .env but does NOT propagate to os.environ, so we do it explicitly.

Call setup_tracing() before any LangChain/LangGraph code runs.
"""
import logging
import os

from langsmith import traceable  # noqa: F401 — re-exported for use across src/ and etl/

from src.config import config

logger = logging.getLogger(__name__)


def setup_tracing(project: str | None = None) -> bool:
    """Configure LangSmith tracing from settings.

    Args:
        project: override the LangSmith project name (e.g. pass "flexllm-test"
                 from conftest.py so test traces never mix with real sessions).

    Returns True if tracing was enabled, False if no API key is present.
    """
    if not config.LANGSMITH_API_KEY:
        logger.info("LangSmith tracing disabled — set LANGSMITH_API_KEY to enable")
        return False

    effective_project = project or config.LANGCHAIN_PROJECT

    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = effective_project
    # Ship traces in a background thread so the main thread is never blocked.
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"
    # Tag every run with the deployment environment for easy filtering in the UI.
    os.environ["LANGCHAIN_TAGS"] = config.ENVIRONMENT

    logger.info(
        "LangSmith tracing enabled — project=%r env=%s",
        effective_project,
        config.ENVIRONMENT,
    )
    return True
