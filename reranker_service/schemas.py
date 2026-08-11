from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .revisions import validate_revision_reference


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID = Field(default_factory=uuid4)
    query: str = Field(min_length=1)
    documents: list[Document] = Field(min_length=1)
    top_n: int | None = Field(None, ge=1)
    return_documents: bool = True
    truncate: bool = True


class Result(BaseModel):
    id: str
    score: float
    normalized_score: float | None = Field(default=None, ge=0, le=1)
    rank: int = Field(ge=1)
    text: str | None = None
    metadata: dict[str, Any] | None = None
    token_count: int | None = None
    truncated: bool = False
    cache_hit: bool = False


class Usage(BaseModel):
    documents_received: int
    documents_scored: int
    cache_hits: int
    latency_ms: int


class RerankResponse(BaseModel):
    request_id: UUID
    model: str
    model_revision: str
    device: str
    requested_revision: str
    resolved_revision: str
    backend: str
    rerank_mode: Literal["pairwise", "listwise"]
    active_provider: str
    results: list[Result]
    usage: Usage


class BatchRequest(BaseModel):
    requests: list[RerankRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    responses: list[RerankResponse]
    total_pairs: int
    latency_ms: int


class RuntimePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int | None = Field(None, ge=1, le=256)
    max_concurrency: int | None = Field(None, ge=1, le=32)
    max_length: int | None = Field(None, ge=64, le=8192)
    dynamic_batching: bool | None = None
    batch_window_ms: int | None = Field(None, ge=0, le=1000)
    max_batch_pairs: int | None = Field(None, ge=1, le=2048)
    request_timeout_seconds: float | None = Field(None, gt=0, le=300)
    default_top_n: int | None = Field(None, ge=1, le=100)

    @model_validator(mode="after")
    def non_empty(self) -> "RuntimePatch":
        if not self.model_fields_set:
            raise ValueError("at least one setting is required")
        return self


class CachePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    ttl_seconds: int | None = Field(None, ge=1, le=2_592_000)

    @model_validator(mode="after")
    def non_empty(self) -> "CachePatch":
        if not self.model_fields_set:
            raise ValueError("at least one cache setting is required")
        return self


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["low_priority", "exclusive"] = "low_priority"
    repetitions: int = Field(5, ge=1, le=100)
    warmup_count: int = Field(1, ge=0, le=20)
    document_count: int = Field(2, ge=1, le=100)
    multilingual: bool = True
    confirm: str | None = None


class ModelCandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    revision: str = Field("main", min_length=1, max_length=200)

    @field_validator("revision")
    @classmethod
    def safe_revision(cls, value: str) -> str:
        return validate_revision_reference(value)


class ModelActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: Literal["ACTIVATE"]
