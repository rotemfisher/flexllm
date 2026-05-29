from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LANGSMITH_API_KEY: str | None = None
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = "flexllm-coach-local"

    DB_PATH: str = "data/personal/running.db"
    QDRANT_PATH: str = "data/qdrant_db"
    QDRANT_COLLECTION: str = "coaching_books"

    MODEL_ID: str = "qwen2.5:32b"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Settings()
