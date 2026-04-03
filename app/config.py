from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/archive"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-4o"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_chunks: int = 3


@lru_cache()
def get_settings() -> Settings:
    return Settings()