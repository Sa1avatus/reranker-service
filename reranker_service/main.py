import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .admin import AdminState
from .batching import DynamicBatcher
from .cache import ScoreCache
from .config import Settings, get_settings
from .errors import ServiceError, service_error_handler
from .metrics import ERRORS, HTTP_DURATION, HTTP_REQUESTS, IN_PROGRESS
from .runtime import ModelRuntime
from .schemas import BatchRequest, BatchResponse, RerankRequest, RerankResponse, RuntimePatch
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        await batcher.start()
        app.state.model_task = asyncio.create_task(runtime.load()) if load_model else None
        if not load_model:
            runtime.model, runtime.ready = "injected", True
        yield
        if app.state.model_task and not app.state.model_task.done():
            app.state.model_task.cancel()
        await batcher.close()
        await cache.close()
        runtime.close()

    app = FastAPI(
        title="Reranker Service",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings, app.state.runtime = cfg, runtime
    app.add_exception_handler(ServiceError, service_error_handler)  # type: ignore[arg-type]

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

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        body = {
            "status": "ready" if runtime.ready else "not_ready",
            "model_ready": runtime.ready,
            "redis": "up" if await cache.ping() else "degraded",
            "error": runtime.load_error,
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
        }

    @app.post("/v1/rerank", response_model=RerankResponse, dependencies=service_deps)
    async def rerank(body: RerankRequest) -> RerankResponse:
        response = await service.rerank(body)
        admin.requests.appendleft(
            {
                "request_id": str(response.request_id),
                "timestamp": time.time(),
                "documents_count": response.usage.documents_received,
                "model": response.model,
                "device": response.device,
                "latency_ms": response.usage.latency_ms,
                "cache_hits": response.usage.cache_hits,
                "status": "success",
            }
        )
        return response

    @app.post("/v1/rerank/batch", response_model=BatchResponse, dependencies=service_deps)
    async def rerank_batch(body: BatchRequest) -> BatchResponse:
        if len(body.requests) > cfg.max_batch_requests:
            raise ServiceError(413, "request_too_large", "too many batch requests")
        started = time.perf_counter()
        responses = await asyncio.gather(*(service.rerank(item) for item in body.requests))
        return BatchResponse(
            responses=responses,
            total_pairs=sum(len(x.documents) for x in body.requests),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    @app.get("/metrics")
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
            },
            "redis": {"available": await cache.ping()},
            "resources": admin.resources(),
            "recent_requests": list(admin.requests)[:10],
        }

    @app.get("/v1/admin/metrics/timeseries", dependencies=admin_deps)
    async def timeseries() -> dict[str, Any]:
        return {"points": [], "generated_at": time.time()}

    @app.get("/v1/admin/runtime", dependencies=admin_deps)
    async def get_runtime() -> dict[str, Any]:
        return admin.runtime | admin.resources() | {"queue_depth": batcher.depth}

    @app.patch("/v1/admin/runtime", dependencies=admin_deps)
    async def patch_runtime(body: RuntimePatch) -> dict[str, Any]:
        return admin.apply(body)

    @app.post("/v1/admin/runtime/validate", dependencies=admin_deps)
    async def validate_runtime(body: RuntimePatch) -> dict[str, Any]:
        return admin.validate(body)

    @app.post("/v1/admin/runtime/reload", dependencies=admin_deps)
    async def reload_runtime() -> dict[str, Any]:
        admin.record("runtime.reload_requested")
        return {"accepted": True, "restart_required": True}

    @app.post("/v1/admin/runtime/rollback", dependencies=admin_deps)
    async def rollback_runtime() -> dict[str, Any]:
        admin.runtime, admin.previous = admin.previous, admin.runtime
        admin.record("runtime.rolled_back")
        return admin.runtime

    @app.get("/v1/admin/models", dependencies=admin_deps)
    async def admin_models() -> dict[str, Any]:
        return await models()

    @app.post("/v1/admin/models/check", dependencies=admin_deps)
    async def check_model(body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name", ""))
        return {"allowed": name in cfg.allowed_models, "name": name}

    @app.post("/v1/admin/models/load", dependencies=admin_deps)
    async def load_candidate(body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name", ""))
        if name not in cfg.allowed_models:
            raise ServiceError(400, "invalid_request", "model not allowlisted")
        admin.record("model.load_requested", {"name": name})
        return {"accepted": True, "restart_required": True}

    @app.post("/v1/admin/models/activate", dependencies=admin_deps)
    async def activate_model(body: dict[str, Any]) -> dict[str, Any]:
        admin.record("model.activate_requested", {"name": body.get("name")})
        return {"accepted": True}

    @app.get("/v1/admin/cache", dependencies=admin_deps)
    async def cache_status() -> dict[str, Any]:
        return {
            "enabled": cfg.cache_enabled,
            "ttl_seconds": cfg.cache_ttl_seconds,
            "redis_available": await cache.ping(),
        }

    @app.patch("/v1/admin/cache", dependencies=admin_deps)
    async def cache_patch(body: dict[str, Any]) -> dict[str, Any]:
        if "enabled" in body:
            cfg.cache_enabled = bool(body["enabled"])
        if "ttl_seconds" in body:
            cfg.cache_ttl_seconds = int(body["ttl_seconds"])
        admin.record("cache.updated", body)
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
    async def create_benchmark(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("mode") == "exclusive" and body.get("confirm") != "EXCLUSIVE":
            raise ServiceError(400, "invalid_request", "exclusive mode confirmation required")
        bid = str(uuid4())
        run = {
            "id": bid,
            "status": "queued",
            "created_at": time.time(),
            "parameters": body,
            "baseline": False,
        }
        admin.benchmarks[bid] = run
        admin.record("benchmark.created", {"id": bid})
        return run

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
        for item in admin.benchmarks.values():
            item["baseline"] = False
        run["baseline"] = True
        admin.record("benchmark.baseline", {"id": benchmark_id})
        return run

    @app.delete("/v1/admin/benchmarks/{benchmark_id}", dependencies=admin_deps)
    async def delete_benchmark(benchmark_id: str) -> dict[str, bool]:
        admin.benchmarks.pop(benchmark_id, None)
        admin.record("benchmark.deleted", {"id": benchmark_id})
        return {"deleted": True}

    @app.get("/v1/admin/requests", dependencies=admin_deps)
    async def requests(page: int = 1, size: int = 50) -> dict[str, Any]:
        items = list(admin.requests)
        start = (page - 1) * size
        return {"items": items[start : start + min(size, 100)], "total": len(items), "page": page}

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
