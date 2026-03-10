from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI (used for embeddings)
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-large"

    # Anthropic (used for response generation)
    anthropic_api_key: str
    anthropic_chat_model: str = "claude-sonnet-4-20250514"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "customer-support-rag"

    # Zendesk (optional)
    zendesk_subdomain: str | None = None
    zendesk_email: str | None = None
    zendesk_api_token: str | None = None

    # Email / IMAP (optional)
    imap_server: str | None = None
    imap_user: str | None = None
    imap_password: str | None = None
    imap_trigger_folder: str = "Generate Draft"

    # API authentication (optional — disabled when None)
    api_key: str | None = None

    # System
    log_level: str = "INFO"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
