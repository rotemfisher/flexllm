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

    # ── WhatsApp bridge ───────────────────────────────────────────────────────
    WHATSAPP_BRIDGE_URL: str = "http://localhost:3001"
    # Shared secret between the Node.js bridge and this API. Must be set in .env.
    WEBHOOK_SECRET: str = "change-me-in-production"
    # Comma-separated E.164 numbers allowed to use the bot, e.g. "+1234567890,+0987654321".
    # Leave empty to allow any number that can reach the bot.
    ALLOWED_NUMBERS: str = ""
    # Minutes of inactivity before a conversation session summary is generated.
    SESSION_TIMEOUT_MINUTES: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_numbers_set(self) -> set[str]:
        """Normalized phone numbers (digits only, no + prefix) from ALLOWED_NUMBERS."""
        if not self.ALLOWED_NUMBERS.strip():
            return set()
        return {n.strip().lstrip("+") for n in self.ALLOWED_NUMBERS.split(",") if n.strip()}


config = Settings()
