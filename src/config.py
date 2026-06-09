from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ── LangSmith tracing (optional) ──────────────────────────────────────────
    LANGSMITH_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "flexllm-coach-local"
    ENVIRONMENT: str = "local"  # local | dev | staging | prod

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://localhost:5432/flexllm"
    QDRANT_PATH: Path = _PROJECT_ROOT / "data" / "qdrant_db"
    # When running the dedicated qdrant container, set QDRANT_URL (e.g.
    # http://qdrant:6333).  If set it takes precedence over QDRANT_PATH.
    QDRANT_URL: str | None = None
    QDRANT_COLLECTION: str = "coaching_books"

    # ── Models ────────────────────────────────────────────────────────────────
    MODEL_ID: str = "qwen2.5:32b"
    # Optional smaller/faster model for RAG query generation.
    # If unset, the main MODEL_ID is reused (no extra VRAM cost).
    # Example: QUERY_MODEL_ID=qwen2.5:3b
    QUERY_MODEL_ID: str | None = None
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Telegram bot ──────────────────────────────────────────────────────────
    # Obtain from @BotFather. Required — app refuses to start if missing.
    TELEGRAM_BOT_TOKEN: str
    # Your numeric Telegram user ID (get it from @userinfobot).
    # The bot ignores every message from any other user ID entirely.
    TELEGRAM_ALLOWED_USER_ID: int

    # ── Scheduler (proactive coaching jobs) ──────────────────────────────────
    # IANA timezone name — used for all scheduled job times.
    SCHEDULER_TIMEZONE: str = "Asia/Jerusalem"

    # ── Legacy Chainlit fields (kept so existing .env files don't break) ──────
    APP_PASSWORD: str = ""
    CHAINLIT_AUTH_SECRET: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = Settings()
