import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model: str = Field("BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    model_revision: str = Field(
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", alias="RERANKER_MODEL_REVISION"
    )
    model_allowlist: str = Field("BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_ALLOWLIST")
    backend: Literal[
        "legacy_cross_encoder", "onnx_pairwise", "jina_listwise", "alibaba_gte"
    ] = Field("onnx_pairwise", alias="RERANKER_BACKEND")
    device: Literal["cpu", "cuda", "auto"] = Field("auto", alias="RERANKER_DEVICE")
    artifact_root: Path = Field(Path("/models/artifacts"), alias="RERANKER_ARTIFACT_ROOT")
    precision: Literal["fp32", "fp16"] = Field("fp32", alias="RERANKER_PRECISION")
    onnx_providers: str = Field(
        "CUDAExecutionProvider,CPUExecutionProvider", alias="RERANKER_ONNX_PROVIDERS"
    )
    cpu_fallback_enabled: bool = Field(True, alias="RERANKER_CPU_FALLBACK_ENABLED")
    trust_remote_code: bool = Field(False, alias="RERANKER_TRUST_REMOTE_CODE")
    remote_code_allowlist: str = Field("", alias="RERANKER_REMOTE_CODE_ALLOWLIST")
    remote_code_revision: str = Field("", alias="RERANKER_REMOTE_CODE_REVISION")
    batch_size: int = Field(16, ge=1, le=256, alias="RERANKER_BATCH_SIZE")
    max_length: int = Field(1024, ge=64, le=8192, alias="RERANKER_MAX_LENGTH")
    max_concurrency: int = Field(2, ge=1, le=32, alias="RERANKER_MAX_CONCURRENCY")
    dynamic_batching: bool = Field(True, alias="RERANKER_DYNAMIC_BATCHING")
    batch_window_ms: int = Field(10, ge=0, le=1000, alias="RERANKER_BATCH_WINDOW_MS")
    max_batch_pairs: int = Field(128, ge=1, le=2048, alias="RERANKER_MAX_BATCH_PAIRS")
    max_batch_tokens: int = Field(16384, ge=1, le=262144, alias="RERANKER_MAX_BATCH_TOKENS")
    max_queue_size: int = Field(256, ge=1, le=10000, alias="RERANKER_MAX_QUEUE_SIZE")
    gpu_max_concurrency: int = Field(1, ge=1, le=8, alias="RERANKER_GPU_MAX_CONCURRENCY")
    cpu_max_concurrency: int = Field(2, ge=1, le=32, alias="RERANKER_CPU_MAX_CONCURRENCY")
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

    @field_validator("backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: object) -> object:
        if value == "onnx":
            return "onnx_pairwise"
        return value

    @field_validator("onnx_providers")
    @classmethod
    def valid_onnx_providers(cls, value: str) -> str:
        providers = [item.strip() for item in value.split(",") if item.strip()]
        supported = {"CUDAExecutionProvider", "CPUExecutionProvider"}
        if not providers or len(providers) != len(set(providers)):
            raise ValueError("ONNX providers must be a non-empty list without duplicates")
        if any(provider not in supported for provider in providers):
            raise ValueError("unsupported ONNX execution provider")
        return ",".join(providers)

    @field_validator("remote_code_revision")
    @classmethod
    def immutable_remote_code_revision(cls, value: str) -> str:
        if value and re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
            raise ValueError("remote code revision must be an immutable 40-character SHA")
        return value.lower()

    @property
    def allowed_models(self) -> set[str]:
        return {x.strip() for x in self.model_allowlist.split(",") if x.strip()}

    @property
    def remote_code_allowed_models(self) -> set[str]:
        return {x.strip() for x in self.remote_code_allowlist.split(",") if x.strip()}

    @property
    def configured_onnx_providers(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.onnx_providers.split(",") if x.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
