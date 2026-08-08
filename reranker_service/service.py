import asyncio
import time
from dataclasses import dataclass

from opentelemetry import trace

from .cache import ScoreCache
from .config import Settings
from .errors import ServiceError
from .metrics import DOCUMENTS_RECEIVED, DOCUMENTS_SCORED, TIMEOUTS, TRUNCATIONS
from .runtime import ModelRuntime
from .schemas import Document, RerankRequest, RerankResponse, Result, Usage

tracer = trace.get_tracer(__name__)


@dataclass
class PreparedDocument:
    document: Document
    text: str
    truncated: bool


class RerankService:
    def __init__(self, settings: Settings, runtime: ModelRuntime, cache: ScoreCache) -> None:
        self.settings, self.runtime, self.cache = settings, runtime, cache

    def _validate(self, request: RerankRequest) -> tuple[str, list[PreparedDocument]]:
        with tracer.start_as_current_span("reranker.validation"):
            if len(request.documents) > self.settings.max_documents:
                raise ServiceError(413, "request_too_large", "too many documents")
            query = request.query
            if len(query) > self.settings.max_query_characters:
                if not request.truncate:
                    raise ServiceError(413, "request_too_large", "query exceeds character limit")
                query = query[: self.settings.max_query_characters]
                TRUNCATIONS.inc()
            prepared = []
            total = len(query)
            for document in request.documents:
                text, was_truncated = document.text, False
                if len(text) > self.settings.max_document_characters:
                    if not request.truncate:
                        raise ServiceError(
                            413, "request_too_large", "document exceeds character limit"
                        )
                    text, was_truncated = text[: self.settings.max_document_characters], True
                    TRUNCATIONS.inc()
                total += len(text)
                prepared.append(PreparedDocument(document, text, was_truncated))
            if total > self.settings.max_total_characters:
                raise ServiceError(413, "request_too_large", "total character limit exceeded")
            return query, prepared

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        if not self.runtime.ready:
            raise ServiceError(503, "model_not_ready", "model is not ready")
        started = time.perf_counter()
        query, prepared = self._validate(request)
        DOCUMENTS_RECEIVED.inc(len(prepared))
        texts = [p.text for p in prepared]
        with tracer.start_as_current_span("reranker.cache_lookup"):
            scores = await self.cache.get_many(query, texts)
        misses = [i for i, score in enumerate(scores) if score is None]
        if misses:
            pairs = [(query, texts[i]) for i in misses]
            try:
                with tracer.start_as_current_span("reranker.inference"):
                    inferred = await asyncio.wait_for(
                        self.runtime.predict(pairs), self.settings.request_timeout_seconds
                    )
            except TimeoutError as exc:
                TIMEOUTS.inc()
                raise ServiceError(504, "inference_timeout", "inference timed out") from exc
            for index, score in zip(misses, inferred, strict=True):
                scores[index] = score
            await self.cache.set_many(query, [texts[i] for i in misses], inferred)
        DOCUMENTS_SCORED.inc(len(prepared))
        indexed: list[tuple[int, float, PreparedDocument]] = []
        for i, prepared_document in enumerate(prepared):
            cached_score = scores[i]
            indexed.append(
                (i, cached_score if cached_score is not None else 0.0, prepared_document)
            )
        with tracer.start_as_current_span("reranker.result_sort"):
            indexed.sort(key=lambda item: -item[1])
        top_n = min(request.top_n or self.settings.default_top_n, len(indexed))
        results = [
            Result(
                id=p.document.id,
                score=score,
                rank=rank,
                text=p.document.text if request.return_documents else None,
                metadata=p.document.metadata if request.return_documents else None,
                token_count=len(p.text.split()),
                truncated=p.truncated,
                cache_hit=i not in misses,
            )
            for rank, (i, score, p) in enumerate(indexed[:top_n], 1)
        ]
        return RerankResponse(
            request_id=request.request_id,
            model=self.settings.model,
            model_revision=self.settings.model_revision,
            device=self.runtime.device,
            results=results,
            usage=Usage(
                documents_received=len(prepared),
                documents_scored=len(prepared),
                cache_hits=len(prepared) - len(misses),
                latency_ms=round((time.perf_counter() - started) * 1000),
            ),
        )
