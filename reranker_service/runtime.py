import asyncio
import math
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
            if self.settings.mock_model:
                self.model = "mock"
            else:
                from sentence_transformers import CrossEncoder

                self.model = await asyncio.to_thread(
                    CrossEncoder,
                    self.settings.model,
                    revision=self.settings.model_revision,
                    device=self.device,
                    max_length=self.settings.max_length,
                    trust_remote_code=False,
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

    def _predict_sync(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if self.settings.mock_model:
            return self._mock_predict(pairs)
        if torch is None:
            raise RuntimeError("PyTorch is required for production inference")
        with torch.inference_mode():
            kwargs = {
                "batch_size": self.settings.batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "activation_fct": torch.nn.Identity(),
            }
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw = self.model.predict(list(pairs), **kwargs)
            else:
                raw = self.model.predict(list(pairs), **kwargs)
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [float(1 / (1 + math.exp(-float(v)))) for v in values]

    async def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        queued = time.perf_counter()
        async with self._semaphore:
            QUEUE_WAIT.observe(time.perf_counter() - queued)
            BATCH_SIZE.observe(len(pairs))
            started = time.perf_counter()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, self._predict_sync, pairs)
            INFERENCE_DURATION.labels(self.settings.model, self.device).observe(
                time.perf_counter() - started
            )
            return result

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
