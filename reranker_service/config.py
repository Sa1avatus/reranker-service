from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model: str = Field("BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    model_revision: str = Field(
        "953dc6f", alias="RERANKER_MODEL_REVISION"
    )
    model_allowlist: str = Field("BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_ALLOWLIST")
    device: Literal["cpu", "cuda", "auto"] = Field("cpu", alias="RERANKER_DEVICE")
    batch_size: int = Field(16, ge=1, le=256, alias="RERANKER_BATCH_SIZE")
    max_length: int = Field(1024, ge=64, le=8192, alias="RERANKER_MAX_LENGTH")
    max_concurrency: int = Field(2, ge=1, le=32, alias="RERANKER_MAX_CONCURRENCY")
    dynamic_batching: bool = Field(True, alias="RERANKER_DYNAMIC_BATCHING")
    batch_window_ms: int = Field(10, ge=0, le=1000, alias="RERANKER_BATCH_WINDOW_MS")
    max_batch_pairs: int = Field(128, ge=1, le=2048, alias="RERANKER_MAX_BATCH_PAIRS")
    cache_enabled: bool = Field(True, alias="RERANKER_CACHE_ENABLED")
    cache_ttl_seconds: int = Field(86400, ge=1, alias="RERANKER_CACHE_TTL_SECONDS")
    redis_url: str = Field("redis://localhost:57379/0", alias="RERANKER_REDIS_URL")
    api_key: SecretStr = Field(..., alias="RERANKER_API_KEY")
    admin_token: SecretStr = Field(..., alias="RERANKER_ADMIN_TOKEN")
    max_documents: int = Field(100, alias="MAX_DOCUMENTS_PER_REQUEST")
    max_query_characters: int = Field(8000, alias="MAX_QUERY_CHARACTERS")
    max_document_characters: int = Field(20000, alias="MAX_DOCUMENT_CHARACTERS")
    max_total_characters: int = Field(500000, alias="MAX_TOTAL_CHARACTERS")
    max_batch_requests: int = Field(32, alias="MAX_BATCH_REQUESTS")
    request_timeout_seconds: float = Field(15, gt=0, alias="REQUEST_TIMEOUT_SECONDS")
    default_top_n: int = Field(10, ge=1, alias="DEFAULT_TOP_N")
    max_body_bytes: int = Field(1_048_576, alias="MAX_BODY_BYTES")
    rate_limit_per_minute: int = Field(120, ge=1, alias="RERANKER_RATE_LIMIT_PER_MINUTE")
    debug_text_logging: bool = Field(False, alias="RERANKER_DEBUG_TEXT_LOGGING")
    mock_model: bool = Field(False, alias="RERANKER_MOCK_MODEL")

    @field_validator("model")
    @classmethod
    def allowed_model(cls, value: str, info: object) -> str:
        return value

    @property
    def allowed_models(self) -> set[str]:
        return {x.strip() for x in self.model_allowlist.split(",") if x.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
