import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .admin import AdminState
from .batching import DynamicBatcher
from .benchmark import BenchmarkRunner
from .cache import ScoreCache
from .config import Settings, get_settings
from .errors import ServiceError, service_error_handler
from .metrics import ERRORS, HTTP_DURATION, HTTP_REQUESTS, IN_PROGRESS
from .runtime import ModelRuntime
from .schemas import (
    BatchRequest,
    BatchResponse,
    BenchmarkSpec,
    CachePatch,
    ModelActivationRequest,
    ModelCandidateRequest,
    RerankRequest,
    RerankResponse,
    RuntimePatch,
)
from .security import Security
from .service import RerankService

structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()]
)
log = structlog.get_logger()


def create_app(settings: Settings | None = None, *, load_model: bool = True) -> FastAPI:
    cfg = settings or get_settings()
    runtime, cache = ModelRuntime(cfg), ScoreCache(cfg)
    batcher = DynamicBatcher(cfg, runtime)
    service = RerankService(cfg, batcher, cache, readiness=runtime, device=runtime.device)
    security, admin = Security(cfg), AdminState(cfg)
    benchmark_runner = BenchmarkRunner(service, admin.benchmarks, admin.record)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        await batcher.start()
        async def load_runtime() -> None:
            try:
                await runtime.load()
            except Exception as exc:
                log.error(
                    "model_load_failed",
                    backend=cfg.backend,
                    model=cfg.model,
                    revision=cfg.model_revision,
                    requested_device=cfg.device,
                    error_type=type(exc).__name__,
                )

        app.state.model_task = asyncio.create_task(load_runtime()) if load_model else None
        if not load_model:
            runtime.model, runtime.ready = "injected", True
        yield
        if app.state.model_task and not app.state.model_task.done():
            app.state.model_task.cancel()
        await benchmark_runner.close()
        await batcher.close()
        await cache.close()
        runtime.close()

    app = FastAPI(
        title="Reranker Service",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        openapi_url=None,
        redoc_url=None,
    )
    app.state.settings, app.state.runtime = cfg, runtime
    app.add_exception_handler(ServiceError, service_error_handler)  # type: ignore[arg-type]

    @app.middleware("http")
    async def validate_backend(request: Request, call_next):  # type: ignore[no-untyped-def]
        selected_backend = request.headers.get("x-backend")
        if (
            request.url.path.startswith("/v1/")
            and selected_backend
            and selected_backend != cfg.backend
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "invalid_backend",
                        "message": f"unknown or unavailable backend: {selected_backend}",
                    }
                },
            )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "details": exc.errors(),
                }
            },
        )

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid4()))
        correlation_id = request.headers.get("x-correlation-id", request_id)
        if (
            request.headers.get("content-length")
            and int(request.headers["content-length"]) > cfg.max_body_bytes
        ):
            return JSONResponse(
                status_code=413,
                content={
                    "error": {"code": "request_too_large", "message": "request body is too large"}
                },
            )
        IN_PROGRESS.inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            response.headers["x-correlation-id"] = correlation_id
            return response
        except Exception:
            ERRORS.labels("500").inc()
            log.exception("unhandled_error", request_id=request_id, path=request.url.path)
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "internal_error", "message": "internal server error"}},
            )
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            endpoint = route.path if route is not None else request.url.path
            HTTP_REQUESTS.labels(endpoint, str(status)).inc()
            HTTP_DURATION.labels(endpoint).observe(elapsed)
            IN_PROGRESS.dec()
            log.info(
                "http_request",
                request_id=request_id,
                correlation_id=correlation_id,
                method=request.method,
                endpoint=endpoint,
                status=status,
                duration_ms=round(elapsed * 1000),
            )

    service_deps = [Depends(security.service_auth), Depends(security.rate_limit)]
    admin_deps = [Depends(security.admin_auth), Depends(security.rate_limit)]

    def record_response(
        response: RerankResponse,
        correlation_id: str,
        request_body: RerankRequest | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "request_id": str(response.request_id),
            "correlation_id": correlation_id,
            "timestamp": time.time(),
            "documents_count": response.usage.documents_received,
            "pairs_count": response.usage.documents_scored,
            "model": response.model,
            "model_revision": response.model_revision,
            "device": response.device,
            "latency_ms": response.usage.latency_ms,
            "cache_hits": response.usage.cache_hits,
            "status": "success",
            "error_code": None,
            "truncation_count": sum(result.truncated for result in response.results),
        }
        if request_body is not None:
            record["query"] = request_body.query[:500]
            record["documents"] = [
                {"id": doc.id, "text": doc.text[:200]}
                for doc in request_body.documents
            ]
            record["results"] = [
                {
                    "id": result.id,
                    "score": result.score,
                    "normalized_score": result.normalized_score,
                    "rank": result.rank,
                    "text": result.text[:200] if result.text else None,
                    "cache_hit": result.cache_hit,
                }
                for result in response.results
            ]
        admin.requests.appendleft(record)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        runtime_state = runtime.state()
        backend_metadata = runtime_state["backend"]
        body = {
            "status": "ready" if runtime.ready else "not_ready",
            "model_ready": runtime.ready,
            "redis": "up" if await cache.ping() else "degraded",
            "error": runtime.load_error,
            "backend": backend_metadata,
            "degraded": runtime_state["degraded"],
            "reason": runtime_state["degraded_reason"],
        }
        return JSONResponse(body, status_code=200 if runtime.ready else 503)

    @app.get("/v1/models", dependencies=service_deps)
    async def models() -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": cfg.model,
                    "revision": cfg.model_revision,
                    "loaded": runtime.ready,
                    "device": runtime.device,
                    "backend": runtime.backend.metadata(runtime.model),
                    "capabilities": runtime.backend.capabilities().as_dict(),
                }
            ]
        }

    @app.get("/v1/models/current", dependencies=service_deps)
    async def current_model() -> dict[str, Any]:
        return {
            "name": cfg.model,
            "revision": cfg.model_revision,
            "device": runtime.device,
            "ready": runtime.ready,
            "max_length": cfg.max_length,
            "backend": runtime.backend.metadata(runtime.model),
            "capabilities": runtime.backend.capabilities().as_dict(),
        }

    @app.post("/v1/rerank", response_model=RerankResponse, dependencies=service_deps)
    async def rerank(body: RerankRequest, request: Request) -> RerankResponse:
        response = await service.rerank(body)
        record_response(
            response,
            request.headers.get("x-correlation-id", str(response.request_id)),
            request_body=body,
        )
        return response

    @app.get("/v1/backends")
    async def backends() -> dict[str, object]:
        """Describe the single backend owned by this API process.

        Multi-backend deployments may replace this response at their routing proxy, but the
        response shape and explicit default remain identical.
        """
        backend_id = cfg.backend
        return {
            "default_backend": backend_id,
            "backends": [
                {
                    "id": backend_id,
                    "name": cfg.model,
                    "backend": cfg.backend,
                    "available": runtime.ready,
                }
            ],
            "model_map": {cfg.model: backend_id},
        }

    @app.post("/v1/rerank/batch", response_model=BatchResponse, dependencies=service_deps)
    async def rerank_batch(body: BatchRequest, request: Request) -> BatchResponse:
        if len(body.requests) > cfg.max_batch_requests:
            raise ServiceError(413, "request_too_large", "too many batch requests")
        started = time.perf_counter()
        responses = await asyncio.gather(*(service.rerank(item) for item in body.requests))
        correlation_id = request.headers.get("x-correlation-id", str(uuid4()))
        for response, req_body in zip(responses, body.requests, strict=True):
            record_response(response, correlation_id, request_body=req_body)
        return BatchResponse(
            responses=responses,
            total_pairs=sum(len(x.documents) for x in body.requests),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    @app.get("/metrics", dependencies=service_deps)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/admin/dashboard", dependencies=admin_deps)
    async def dashboard() -> dict[str, Any]:
        return {
            "model": {
                "name": cfg.model,
                "revision": cfg.model_revision,
                "ready": runtime.ready,
                "device": runtime.device,
                "backend": runtime.backend.metadata(runtime.model),
                "capabilities": runtime.backend.capabilities().as_dict(),
            },
            "redis": {"available": await cache.ping()},
            "resources": admin.resources(),
            "recent_requests": list(admin.requests)[:10],
        }

    @app.post("/v1/admin/rerank", response_model=RerankResponse, dependencies=admin_deps)
    async def admin_rerank(body: RerankRequest, request: Request) -> RerankResponse:
        return await rerank(body, request)

    @app.post("/v1/admin/rerank/batch", response_model=BatchResponse, dependencies=admin_deps)
    async def admin_rerank_batch(body: BatchRequest, request: Request) -> BatchResponse:
        return await rerank_batch(body, request)

    @app.get("/v1/admin/metrics/timeseries", dependencies=admin_deps)
    async def timeseries(
        period_seconds: int = Query(3600, ge=60, le=604800),
        bucket_seconds: int = Query(60, ge=10, le=86400),
    ) -> dict[str, Any]:
        return {
            "points": admin.timeseries(period_seconds, bucket_seconds),
            "period_seconds": period_seconds,
            "bucket_seconds": bucket_seconds,
            "generated_at": time.time(),
        }

    @app.get("/v1/admin/runtime", dependencies=admin_deps)
    async def get_runtime() -> dict[str, Any]:
        return admin.runtime | admin.resources() | {"queue_depth": batcher.depth}

    @app.patch("/v1/admin/runtime", dependencies=admin_deps)
    async def patch_runtime(body: RuntimePatch) -> dict[str, Any]:
        result = admin.apply(body)
        updates = body.model_dump(exclude_none=True)
        for name, value in updates.items():
            setattr(cfg, name, value)
        runtime.reconfigure(
            max_concurrency=cfg.max_concurrency,
            max_length=cfg.max_length,
        )
        await batcher.reconfigure()
        return result

    @app.post("/v1/admin/runtime/validate", dependencies=admin_deps)
    async def validate_runtime(body: RuntimePatch) -> dict[str, Any]:
        return admin.validate(body)

    @app.post("/v1/admin/runtime/reload", dependencies=admin_deps)
    async def reload_runtime() -> dict[str, Any]:
        admin.record("runtime.reload_requested")
        return {"accepted": True, "restart_required": True}

    @app.post("/v1/admin/runtime/rollback", dependencies=admin_deps)
    async def rollback_runtime() -> dict[str, Any]:
        if runtime.previous is not None:
            state = await runtime.rollback()
            admin.record(
                "model.rolled_back", {"name": state["name"], "revision": state["revision"]}
            )
            return state
        admin.runtime, admin.previous = admin.previous, admin.runtime
        admin.record("runtime.rolled_back")
        return admin.runtime

    @app.get("/v1/admin/models", dependencies=admin_deps)
    async def admin_models() -> dict[str, Any]:
        latencies = [
            float(item["latency_ms"])
            for item in admin.requests
            if item["model"] == cfg.model and item["status"] == "success"
        ]
        return {
            "active_model": cfg.model,
            "candidate": runtime.candidate_info,
            "rollback_available": runtime.previous is not None,
            "models": [
                {
                    "name": name,
                    "revision": cfg.model_revision if name == cfg.model else None,
                    "status": "ready" if name == cfg.model and runtime.ready else "available",
                    "loaded": name == cfg.model and runtime.ready,
                    "device_support": ["cpu", "cuda"] if runtime.device == "cuda" else ["cpu"],
                    "max_length": cfg.max_length,
                    "estimated_memory_bytes": 2_500_000_000,
                    "last_loaded": runtime.loaded_at if name == cfg.model else None,
                    "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                }
                for name in sorted(cfg.allowed_models)
            ],
        }

    @app.post("/v1/admin/models/check", dependencies=admin_deps)
    async def check_model(body: ModelCandidateRequest) -> dict[str, Any]:
        return runtime.check_candidate(body.name, body.revision)

    @app.post("/v1/admin/models/load", dependencies=admin_deps)
    async def load_candidate(body: ModelCandidateRequest) -> dict[str, Any]:
        try:
            result = await runtime.load_candidate(body.name, body.revision)
        except ValueError as exc:
            raise ServiceError(400, "invalid_request", str(exc)) from exc
        admin.record(
            "model.candidate_loaded",
            {"name": body.name, "revision": body.revision,
             "controlled_restart_required": result["controlled_restart_required"]},
        )
        return result

    @app.post("/v1/admin/models/activate", dependencies=admin_deps)
    async def activate_model(_: ModelActivationRequest) -> dict[str, Any]:
        try:
            state = await runtime.activate_candidate()
        except ValueError as exc:
            raise ServiceError(409, "candidate_not_ready", str(exc)) from exc
        admin.record("model.activated", {"name": state["name"], "revision": state["revision"]})
        return state

    @app.get("/v1/admin/cache", dependencies=admin_deps)
    async def cache_status() -> dict[str, Any]:
        return {
            "enabled": cfg.cache_enabled,
            "ttl_seconds": cfg.cache_ttl_seconds,
            "redis_available": await cache.ping(),
        }

    @app.patch("/v1/admin/cache", dependencies=admin_deps)
    async def cache_patch(body: CachePatch) -> dict[str, Any]:
        updates = body.model_dump(exclude_none=True)
        if "enabled" in updates:
            cfg.cache_enabled = updates["enabled"]
        if "ttl_seconds" in updates:
            cfg.cache_ttl_seconds = updates["ttl_seconds"]
        admin.record("cache.updated", updates)
        return await cache_status()

    @app.post("/v1/admin/cache/clear", dependencies=admin_deps)
    async def cache_clear(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("confirm") != "CLEAR":
            raise ServiceError(400, "invalid_request", "confirmation required")
        deleted = await cache.clear(bool(body.get("model_only")))
        admin.record("cache.cleared", {"deleted": deleted})
        return {"deleted": deleted}

    @app.post("/v1/admin/cache/test", dependencies=admin_deps)
    async def cache_test() -> dict[str, Any]:
        return {"available": await cache.ping()}

    @app.post("/v1/admin/benchmarks", dependencies=admin_deps)
    async def create_benchmark(body: BenchmarkSpec) -> dict[str, Any]:
        if body.mode == "exclusive" and body.confirm != "EXCLUSIVE":
            raise ServiceError(400, "invalid_request", "exclusive mode confirmation required")
        if not runtime.ready:
            raise ServiceError(503, "model_not_ready", "model is not ready")
        return benchmark_runner.start(body)

    @app.get("/v1/admin/benchmarks", dependencies=admin_deps)
    async def benchmarks() -> dict[str, Any]:
        return {"items": list(admin.benchmarks.values())}

    @app.get("/v1/admin/benchmarks/{benchmark_id}", dependencies=admin_deps)
    async def benchmark(benchmark_id: str) -> dict[str, Any]:
        if benchmark_id not in admin.benchmarks:
            raise ServiceError(404, "not_found", "benchmark not found")
        return admin.benchmarks[benchmark_id]

    @app.post("/v1/admin/benchmarks/{benchmark_id}/baseline", dependencies=admin_deps)
    async def baseline(benchmark_id: str) -> dict[str, Any]:
        run = await benchmark(benchmark_id)
        if run["status"] != "completed":
            raise ServiceError(409, "benchmark_not_complete", "benchmark is not complete")
        for item in admin.benchmarks.values():
            item["baseline"] = False
        run["baseline"] = True
        admin.record("benchmark.baseline", {"id": benchmark_id})
        return run

    @app.delete("/v1/admin/benchmarks/{benchmark_id}", dependencies=admin_deps)
    async def delete_benchmark(benchmark_id: str) -> dict[str, bool]:
        if not await benchmark_runner.delete(benchmark_id):
            raise ServiceError(404, "not_found", "benchmark not found")
        return {"deleted": True}

    @app.get("/v1/admin/requests", dependencies=admin_deps)
    async def requests(
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        status: str | None = None,
        model_name: str | None = None,
        min_latency_ms: float | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        items = [
            item
            for item in admin.requests
            if (status is None or item["status"] == status)
            and (model_name is None or item["model"] == model_name)
            and (min_latency_ms is None or float(item["latency_ms"]) >= min_latency_ms)
        ]
        start = (page - 1) * size
        return {
            "items": items[start : start + size],
            "total": len(items),
            "page": page,
            "size": size,
        }

    @app.get("/v1/admin/requests/{request_id}", dependencies=admin_deps)
    async def get_request(request_id: str) -> dict[str, Any]:
        item = next((x for x in admin.requests if x["request_id"] == request_id), None)
        if not item:
            raise ServiceError(404, "not_found", "request not found")
        return item

    @app.post("/v1/admin/requests/{request_id}/repeat", dependencies=admin_deps)
    async def repeat_request(request_id: str) -> dict[str, Any]:
        await get_request(request_id)
        raise ServiceError(409, "payload_not_retained", "request payload was not retained")

    @app.get("/v1/admin/system/health", dependencies=admin_deps)
    async def system_health() -> dict[str, Any]:
        return {
            "api": "up",
            "model": "up" if runtime.ready else "down",
            "redis": "up" if await cache.ping() else "degraded",
            "warmup": runtime.ready,
        }

    @app.get("/v1/admin/system/resources", dependencies=admin_deps)
    async def resources() -> dict[str, Any]:
        return admin.resources()

    @app.get("/v1/admin/audit-log", dependencies=admin_deps)
    async def audit_log(page: int = 1, size: int = 50) -> dict[str, Any]:
        items = list(admin.audit)
        start = (page - 1) * size
        return {"items": items[start : start + min(size, 100)], "total": len(items), "page": page}

    FastAPIInstrumentor.instrument_app(app)
    return app


app = create_app()
