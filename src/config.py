from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ── LangSmith tracing (optional) ──────────────────────────────────────────
    LANGSMITH_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "flexllm-coach-local"
    ENVIRONMENT: str = "local"  # local | dev | staging | prod

    # ── Data paths ────────────────────────────────────────────────────────────
    DB_PATH: str = str(_PROJECT_ROOT / "data" / "personal" / "running.db")
    QDRANT_PATH: str = str(_PROJECT_ROOT / "data" / "qdrant_db")
    QDRANT_COLLECTION: str = "coaching_books"

    # ── Models ────────────────────────────────────────────────────────────────
    MODEL_ID: str = "qwen2.5:32b"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Auth (Chainlit password gate) ─────────────────────────────────────────
    # Set APP_PASSWORD in .env. Anyone with the Cloudflare URL must know it.
    APP_PASSWORD: str = "change-me-in-env"
    # Must be a stable random secret — sessions invalidate if this changes.
    # Generate with: openssl rand -hex 32
    CHAINLIT_AUTH_SECRET: str = "change-me-in-env"

    # ── Cloudflare Tunnel (optional) ──────────────────────────────────────────
    # Leave blank → ephemeral *.trycloudflare.com URL printed to logs.
    # Set a token (from Cloudflare dashboard) → persistent custom domain.
    CLOUDFLARE_TUNNEL_TOKEN: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = Settings()
