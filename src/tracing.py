"""
LangSmith tracing activation for FlexLLM.

LangChain/LangGraph read tracing settings from os.environ. Pydantic Settings
reads from .env but does NOT propagate to os.environ, so we do it explicitly.

Call setup_tracing() before any LangChain/LangGraph code runs.
"""
import os

from src.config import config


def setup_tracing() -> bool:
    """Set os.environ tracing vars when LANGSMITH_API_KEY is configured.

    Returns True if tracing was enabled, False if no API key is present.
    """
    if not config.LANGSMITH_API_KEY:
        return False
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
    return True
