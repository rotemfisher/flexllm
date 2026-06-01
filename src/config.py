from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    LANGSMITH_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "flexllm-coach-local"
    ENVIRONMENT: str = "local"  # local | dev | staging | prod — tags traces in LangSmith

    DB_PATH: str = str(_PROJECT_ROOT / "data" / "personal" / "running.db")
    QDRANT_PATH: str = str(_PROJECT_ROOT / "data" / "qdrant_db")
    QDRANT_COLLECTION: str = "coaching_books"

    MODEL_ID: str = "qwen2.5:32b"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Settings()
