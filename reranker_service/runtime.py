import asyncio
import math
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil

try:
    import torch
except ImportError:  # Unit-test/minimal control-plane installations may omit the model runtime.
    torch = None

from .config import Settings
from .metrics import BATCH_SIZE, INFERENCE_DURATION, MODEL_LOAD, MODEL_READY, QUEUE_WAIT


class ModelRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any = None
        self.ready = False
        self.loaded_at: float | None = None
        self.load_error: str | None = None
        self.device = self._resolve_device(settings.device)
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrency, thread_name_prefix="reranker"
        )
        self._model_lock = asyncio.Lock()
        self.candidate: Any = None
        self.candidate_info: dict[str, Any] | None = None
        self.previous: tuple[Any, str, str, float | None] | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but unavailable")
        return requested

    async def load(self) -> None:
        started = time.perf_counter()
        try:
            self.model = await self._load_model(
                self.settings.model, self.settings.model_revision, self.settings.max_length
            )
            await self.predict([("warmup", "warmup")])
            self.ready, self.loaded_at = True, time.time()
            MODEL_READY.labels(self.settings.model, self.device).set(1)
        except Exception as exc:
            self.load_error = str(exc)
            MODEL_READY.labels(self.settings.model, self.device).set(0)
            raise
        finally:
            MODEL_LOAD.labels(self.settings.model, self.device).observe(
                time.perf_counter() - started
            )

    def _mock_predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        for query, document in pairs:
            q = set(re.findall(r"\w+", query.casefold()))
            d = set(re.findall(r"\w+", document.casefold()))
            overlap = len(q & d) / max(len(q), 1)
            scores.append(1 / (1 + math.exp(-(overlap * 8 - 3))))
        return scores

    async def _load_model(self, name: str, revision: str, max_length: int) -> Any:
        if self.settings.mock_model:
            return f"mock:{name}@{revision}"
        from sentence_transformers import CrossEncoder

        return await asyncio.to_thread(
            CrossEncoder,
            name,
            revision=revision,
            device=self.device,
            max_length=max_length,
            trust_remote_code=False,
        )

    def _predict_sync(
        self, model: Any, pairs: Sequence[tuple[str, str]]
    ) -> list[float]:
        if self.settings.mock_model:
            return self._mock_predict(pairs)
        if torch is None:
            raise RuntimeError("PyTorch is required for production inference")
        with torch.inference_mode():
            kwargs = {
                "batch_size": self.settings.batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "activation_fn": torch.nn.Identity(),
            }
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw = model.predict(list(pairs), **kwargs)
            else:
                raw = model.predict(list(pairs), **kwargs)
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [float(1 / (1 + math.exp(-float(v)))) for v in values]

    async def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        queued = time.perf_counter()
        async with self._semaphore:
            QUEUE_WAIT.observe(time.perf_counter() - queued)
            BATCH_SIZE.observe(len(pairs))
            started = time.perf_counter()
            loop = asyncio.get_running_loop()
            model = self.model
            result = await loop.run_in_executor(self._executor, self._predict_sync, model, pairs)
            INFERENCE_DURATION.labels(self.settings.model, self.device).observe(
                time.perf_counter() - started
            )
            return result

    def check_candidate(self, name: str, revision: str) -> dict[str, Any]:
        revision_valid = bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", revision))
        estimated_bytes = 3_500_000_000 if self.device == "cpu" else 3_000_000_000
        available_bytes = psutil.virtual_memory().available
        allowed = name in self.settings.allowed_models
        can_parallel_load = self.settings.mock_model or available_bytes >= estimated_bytes
        return {
            "name": name,
            "revision": revision,
            "allowed": allowed,
            "revision_valid": revision_valid,
            "device": self.device,
            "available_memory_bytes": available_bytes,
            "estimated_additional_memory_bytes": estimated_bytes,
            "can_parallel_load": can_parallel_load,
            "controlled_restart_required": not can_parallel_load,
            "valid": allowed and revision_valid,
        }

    async def load_candidate(self, name: str, revision: str) -> dict[str, Any]:
        check = self.check_candidate(name, revision)
        if not check["valid"]:
            raise ValueError("candidate model or revision is not allowed")
        if not check["can_parallel_load"]:
            return check
        async with self._model_lock:
            started = time.perf_counter()
            candidate = await self._load_model(name, revision, self.settings.max_length)
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                self._executor,
                self._predict_sync,
                candidate,
                [("Kubernetes experience", "Production Kubernetes operations")],
            )
            if len(scores) != 1 or not 0 <= scores[0] <= 1:
                raise RuntimeError("candidate test rerank failed")
            self.candidate = candidate
            self.candidate_info = {
                **check,
                "status": "ready",
                "warmup_complete": True,
                "test_score": scores[0],
                "load_seconds": time.perf_counter() - started,
                "loaded_at": time.time(),
            }
            return self.candidate_info

    async def activate_candidate(self) -> dict[str, Any]:
        async with self._model_lock:
            if self.candidate is None or self.candidate_info is None:
                raise ValueError("no candidate model is ready")
            old_name, old_revision = self.settings.model, self.settings.model_revision
            self.previous = (self.model, old_name, old_revision, self.loaded_at)
            self.model = self.candidate
            self.settings.model = str(self.candidate_info["name"])
            self.settings.model_revision = str(self.candidate_info["revision"])
            self.loaded_at = float(self.candidate_info["loaded_at"])
            self.candidate, self.candidate_info = None, None
            MODEL_READY.labels(old_name, self.device).set(0)
            MODEL_READY.labels(self.settings.model, self.device).set(1)
            return self.state()

    async def rollback(self) -> dict[str, Any]:
        async with self._model_lock:
            if self.previous is None:
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
        }

    def reconfigure(self, *, max_concurrency: int, max_length: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.settings.max_concurrency = max_concurrency
        self.settings.max_length = max_length
        if self.model is not None and not isinstance(self.model, str):
            self.model.max_length = max_length

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
