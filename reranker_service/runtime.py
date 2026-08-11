import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil
import structlog

from .backends import BackendRegistry, RerankerBackend, default_backend_registry
from .backends.base import RerankPair
from .config import Settings
from .metrics import (
    BATCH_SIZE,
    INFERENCE_DURATION,
    MODEL_LOAD,
    MODEL_READY,
    PROVIDER_ACTIVE,
    PROVIDER_FALLBACKS,
    QUEUE_WAIT,
)
from .revisions import resolve_immutable_revision, validate_revision_reference

log = structlog.get_logger()


class ModelRuntime:
    def __init__(self, settings: Settings, registry: BackendRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or default_backend_registry()
        self.backend: RerankerBackend = self.registry.create(settings.backend, settings)
        self.model: Any = None
        self.ready = False
        self.loaded_at: float | None = None
        self.load_error: str | None = None
        self.device = self.backend.device
        self.effective_max_concurrency = settings.max_concurrency
        self._semaphore = asyncio.Semaphore(self.effective_max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max(
                settings.max_concurrency,
                settings.gpu_max_concurrency,
                settings.cpu_max_concurrency,
            ),
            thread_name_prefix="reranker",
        )
        self._model_lock = asyncio.Lock()
        self.candidate: Any = None
        self.candidate_info: dict[str, Any] | None = None
        self.previous: tuple[Any, str, str, float | None] | None = None
        self._active_provider: str | None = None
        self._last_fallback: tuple[str, str] | None = None
        self.operation: dict[str, Any] = {
            "type": None,
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def _set_operation(
        self,
        operation_type: str,
        status: str,
        *,
        started_at: float,
        error: str | None = None,
    ) -> None:
        self.operation = {
            "type": operation_type,
            "status": status,
            "started_at": started_at,
            "finished_at": None if status == "running" else time.time(),
            "error": error,
        }

    async def load(self) -> None:
        started = time.perf_counter()
        operation_started = time.time()
        self._set_operation("initial_load", "running", started_at=operation_started)
        try:
            self.model = await self.backend.load(
                self.settings.model, self.settings.model_revision, self.settings.max_length
            )
            self._sync_active_backend_state()
            await self._warmup(self.model)
            self.ready, self.loaded_at = True, time.time()
            metadata = self.backend.metadata(self.model)
            log.info(
                "model_ready",
                backend=self.backend.name,
                model=self.settings.model,
                requested_revision=metadata.get(
                    "requested_revision", self.settings.model_revision
                ),
                resolved_revision=metadata.get("resolved_revision", self.settings.model_revision),
                requested_device=self.settings.device,
                active_provider=metadata.get("active_provider", self.device),
                fallback_provider=metadata.get("fallback_provider"),
                precision=metadata.get("precision"),
            )
            MODEL_READY.labels(self.settings.model, self.device).set(1)
            self.load_error = None
            self._set_operation("initial_load", "completed", started_at=operation_started)
        except Exception as exc:
            self.load_error = str(exc)
            MODEL_READY.labels(self.settings.model, self.device).set(0)
            self._set_operation(
                "initial_load",
                "error",
                started_at=operation_started,
                error=str(exc),
            )
            raise
        finally:
            MODEL_LOAD.labels(self.settings.model, self.device).observe(
                time.perf_counter() - started
            )

    async def _warmup(self, model: Any) -> None:
        BATCH_SIZE.observe(1)
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.backend.warmup, model)
        INFERENCE_DURATION.labels(self.settings.model, self.device).observe(
            time.perf_counter() - started
        )

    async def predict(self, pairs: list[RerankPair]) -> list[float]:
        queued = time.perf_counter()
        async with self._semaphore:
            QUEUE_WAIT.observe(time.perf_counter() - queued)
            BATCH_SIZE.observe(len(pairs))
            started = time.perf_counter()
            loop = asyncio.get_running_loop()
            model = self.model
            result = await loop.run_in_executor(self._executor, self.backend.rerank, model, pairs)
            self._sync_active_backend_state()
            INFERENCE_DURATION.labels(self.settings.model, self.device).observe(
                time.perf_counter() - started
            )
            return result

    def check_candidate(self, name: str, revision: str) -> dict[str, Any]:
        try:
            validate_revision_reference(revision)
            revision_valid = True
        except ValueError:
            revision_valid = False
        resolved_revision: str | None = None
        resolution_error: str | None = None
        if revision_valid and name in self.settings.allowed_models:
            try:
                resolved_revision = (
                    revision.lower()
                    if self.settings.mock_model
                    else resolve_immutable_revision(name, revision)
                )
            except Exception as exc:
                resolution_error = type(exc).__name__
        estimated_bytes = 3_500_000_000 if self.device == "cpu" else 3_000_000_000
        available_bytes = psutil.virtual_memory().available
        allowed = name in self.settings.allowed_models
        can_parallel_load = self.settings.mock_model or available_bytes >= estimated_bytes
        backend_requirements = self.backend.candidate_requirements(
            name, resolved_revision or revision
        )
        if self.device == "cuda" and backend_requirements.get("artifact_available"):
            can_parallel_load = can_parallel_load and bool(
                backend_requirements.get("gpu_parallel_load_supported")
            )
        return {
            "name": name,
            "revision": revision,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "revision_resolution_error": resolution_error,
            "allowed": allowed,
            "revision_valid": revision_valid,
            "device": self.device,
            "available_memory_bytes": available_bytes,
            "estimated_additional_memory_bytes": estimated_bytes,
            "can_parallel_load": can_parallel_load,
            "controlled_restart_required": not can_parallel_load,
            "valid": allowed and revision_valid and resolved_revision is not None,
            "backend_requirements": backend_requirements,
        }

    async def load_candidate(self, name: str, revision: str) -> dict[str, Any]:
        operation_started = time.time()
        self._set_operation("candidate_load", "running", started_at=operation_started)
        try:
            check = self.check_candidate(name, revision)
            if not check["valid"]:
                raise ValueError("candidate model or revision is not allowed")
            if not check["can_parallel_load"]:
                self._set_operation(
                    "candidate_load", "completed", started_at=operation_started
                )
                return check
            async with self._model_lock:
                started = time.perf_counter()
                resolved_revision = str(check["resolved_revision"])
                candidate = await self.backend.load(
                    name, resolved_revision, self.settings.max_length
                )
                try:
                    loop = asyncio.get_running_loop()
                    scores = await loop.run_in_executor(
                        self._executor,
                        self.backend.rerank,
                        candidate,
                        [("Kubernetes experience", "Production Kubernetes operations")],
                    )
                    if len(scores) != 1 or not math.isfinite(scores[0]):
                        raise RuntimeError("candidate test rerank failed")
                except Exception:
                    self.backend.unload(candidate)
                    raise
                self.candidate = candidate
                self.candidate_info = {
                    **check,
                    "status": "ready",
                    "warmup_complete": True,
                    "test_score": scores[0],
                    "load_seconds": time.perf_counter() - started,
                    "loaded_at": time.time(),
                    "backend_metadata": self.backend.metadata(candidate),
                }
                self._set_operation(
                    "candidate_load", "completed", started_at=operation_started
                )
                return self.candidate_info
        except Exception as exc:
            self._set_operation(
                "candidate_load",
                "error",
                started_at=operation_started,
                error=str(exc),
            )
            raise

    async def activate_candidate(self) -> dict[str, Any]:
        operation_started = time.time()
        self._set_operation("candidate_activate", "running", started_at=operation_started)
        async with self._model_lock:
            if self.candidate is None or self.candidate_info is None:
                self._set_operation(
                    "candidate_activate",
                    "error",
                    started_at=operation_started,
                    error="no candidate model is ready",
                )
                raise ValueError("no candidate model is ready")
            old_name, old_revision = self.settings.model, self.settings.model_revision
            self.previous = (self.model, old_name, old_revision, self.loaded_at)
            self.model = self.candidate
            self.settings.model = str(self.candidate_info["name"])
            self.settings.model_revision = str(self.candidate_info["resolved_revision"])
            self.loaded_at = float(self.candidate_info["loaded_at"])
            self.candidate, self.candidate_info = None, None
            self._sync_active_backend_state()
            MODEL_READY.labels(old_name, self.device).set(0)
            MODEL_READY.labels(self.settings.model, self.device).set(1)
            self._set_operation(
                "candidate_activate", "completed", started_at=operation_started
            )
            return self.state()

    async def rollback(self) -> dict[str, Any]:
        operation_started = time.time()
        self._set_operation("rollback", "running", started_at=operation_started)
        async with self._model_lock:
            if self.previous is None:
                self._set_operation(
                    "rollback",
                    "error",
                    started_at=operation_started,
                    error="no previous model is available",
                )
                raise ValueError("no previous model is available")
            current = (
                self.model,
                self.settings.model,
                self.settings.model_revision,
                self.loaded_at,
            )
            (
                self.model,
                self.settings.model,
                self.settings.model_revision,
                self.loaded_at,
            ) = self.previous
            self.previous = current
            self._sync_active_backend_state()
            self._set_operation("rollback", "completed", started_at=operation_started)
            return self.state()

    def state(self) -> dict[str, Any]:
        return {
            "name": self.settings.model,
            "revision": self.settings.model_revision,
            "device": self.device,
            "ready": self.ready,
            "loaded_at": self.loaded_at,
            "candidate": self.candidate_info,
            "rollback_available": self.previous is not None,
            "backend": self.backend.metadata(self.model),
            "degraded": bool(self.backend.metadata(self.model).get("fallback_reason")),
            "degraded_reason": self.backend.metadata(self.model).get("fallback_reason"),
            "max_concurrent_inference": self.effective_max_concurrency,
            "operation": self.operation,
        }

    def reconfigure(self, *, max_concurrency: int, max_length: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.settings.max_concurrency = max_concurrency
        self.settings.max_length = max_length
        self._configure_concurrency()
        if self.model is not None:
            self.backend.reconfigure(self.model, max_length)

    def _sync_active_backend_state(self) -> None:
        metadata = self.backend.metadata(self.model)
        active_provider = metadata.get("active_provider")
        if isinstance(active_provider, str):
            if self._active_provider is not None:
                PROVIDER_ACTIVE.labels(self.backend.name, self._active_provider).set(0)
            PROVIDER_ACTIVE.labels(self.backend.name, active_provider).set(1)
            self._active_provider = active_provider
            provider_devices = {
                "CUDAExecutionProvider": "cuda",
                "CPUExecutionProvider": "cpu",
            }
            self.device = provider_devices.get(active_provider, active_provider)
            self._configure_concurrency()
        fallback_reason = metadata.get("fallback_reason")
        fallback_provider = metadata.get("fallback_provider")
        if isinstance(fallback_reason, str) and isinstance(fallback_provider, str):
            fallback = (fallback_provider, fallback_reason)
            if fallback != self._last_fallback:
                PROVIDER_FALLBACKS.labels(
                    self.backend.name, fallback_provider, fallback_reason
                ).inc()
                self._last_fallback = fallback

    def _configure_concurrency(self) -> None:
        provider_limit = (
            self.settings.gpu_max_concurrency
            if self.device == "cuda"
            else self.settings.cpu_max_concurrency
        )
        self.effective_max_concurrency = min(self.settings.max_concurrency, provider_limit)
        self._semaphore = asyncio.Semaphore(self.effective_max_concurrency)

    def close(self) -> None:
        models = [self.model, self.candidate]
        if self.previous is not None:
            models.append(self.previous[0])
        seen: set[int] = set()
        for model in models:
            if model is not None and id(model) not in seen:
                seen.add(id(model))
                self.backend.unload(model)
        self.model = self.candidate = None
        self.previous = None
        self._executor.shutdown(wait=False, cancel_futures=True)
