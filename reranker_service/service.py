import asyncio
import time
from dataclasses import dataclass

from opentelemetry import trace

from .batching import PairPredictor
from .cache import ScoreCache
from .config import Settings
from .errors import ServiceError
from .metrics import DOCUMENTS_RECEIVED, DOCUMENTS_SCORED, TIMEOUTS, TRUNCATIONS
from .schemas import Document, RerankRequest, RerankResponse, Result, Usage

tracer = trace.get_tracer(__name__)


@dataclass
class PreparedDocument:
    document: Document
    text: str
    truncated: bool


class RerankService:
    def __init__(
        self,
        settings: Settings,
        runtime: PairPredictor,
        cache: ScoreCache,
        *,
        readiness: object,
        device: str,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.cache = cache
        self.readiness = readiness
        self.device = device

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
        if not bool(getattr(self.readiness, "ready", False)):
            raise ServiceError(503, "model_not_ready", "model is not ready")
        started = time.perf_counter()
        backend = getattr(self.readiness, "backend", None)
        active_model = getattr(self.readiness, "model", None)
        capabilities = backend.capabilities() if backend is not None else None
        if capabilities is not None and len(request.documents) > capabilities.max_documents:
            raise ServiceError(413, "request_too_large", "too many documents for active backend")
        listwise = capabilities is not None and capabilities.rerank_mode == "listwise"
        query, prepared = self._validate(request)
        DOCUMENTS_RECEIVED.inc(len(prepared))
        texts = [p.text for p in prepared]
        if listwise:
            scores: list[float | None] = [None] * len(texts)
        else:
            with tracer.start_as_current_span("reranker.cache_lookup"):
                scores = await self.cache.get_many(query, texts)
        misses = [i for i, score in enumerate(scores) if score is None]
        if misses:
            pairs = [(query, texts[i]) for i in misses]
            predictor = self.readiness if listwise else self.runtime
            predict = getattr(predictor, "predict", None)
            if not callable(predict):
                raise ServiceError(503, "model_not_ready", "active backend cannot run inference")
            try:
                with tracer.start_as_current_span("reranker.inference"):
                    inferred = await asyncio.wait_for(
                        predict(pairs), self.settings.request_timeout_seconds
                    )
            except TimeoutError as exc:
                TIMEOUTS.inc()
                raise ServiceError(504, "inference_timeout", "inference timed out") from exc
            for index, score in zip(misses, inferred, strict=True):
                scores[index] = score
            if not listwise:
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
        backend_metadata = backend.metadata(active_model) if backend is not None else {}
        resolved_revision = str(
            backend_metadata.get("resolved_revision", self.settings.model_revision)
        )
        requested_revision = str(
            backend_metadata.get("requested_revision", self.settings.model_revision)
        )
        active_device = str(getattr(self.readiness, "device", self.device))
        results = [
            Result(
                id=p.document.id,
                score=score,
                normalized_score=(
                    backend.normalized_score(active_model, score)
                    if backend is not None
                    else None
                ),
                rank=rank,
                text=p.document.text if request.return_documents else None,
                metadata=p.document.metadata,
                token_count=len(p.text.split()),
                truncated=p.truncated,
                cache_hit=i not in misses,
            )
            for rank, (i, score, p) in enumerate(indexed[:top_n], 1)
        ]
        return RerankResponse(
            request_id=request.request_id,
            model=self.settings.model,
            model_revision=resolved_revision,
            device=active_device,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            backend=str(backend_metadata.get("backend", self.settings.backend)),
            rerank_mode=capabilities.rerank_mode if capabilities is not None else "pairwise",
            active_provider=str(backend_metadata.get("active_provider", active_device)),
            results=results,
            usage=Usage(
                documents_received=len(prepared),
                documents_scored=len(prepared),
                cache_hits=len(prepared) - len(misses),
                latency_ms=round((time.perf_counter() - started) * 1000),
            ),
        )
