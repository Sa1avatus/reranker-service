import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings
from .errors import ServiceError
from .metrics import QUEUE_WAIT


class PairPredictor(Protocol):
    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@dataclass
class BatchJob:
    pairs: list[tuple[str, str]]
    future: asyncio.Future[list[float]]
    enqueued_at: float = field(default_factory=time.perf_counter)
    cursor: int = 0
    scores: list[float] = field(default_factory=list)


class DynamicBatcher:
    """Combines pairs from concurrent requests without mixing their results."""

    def __init__(self, settings: Settings, predictor: PairPredictor) -> None:
        self.settings = settings
        self.predictor = predictor
        self.queue: asyncio.Queue[BatchJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._worker: asyncio.Task[None] | None = None

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def start(self) -> None:
        if self.settings.dynamic_batching and self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="reranker-micro-batcher")

    async def reconfigure(self) -> None:
        if self.settings.dynamic_batching and self._worker is None:
            await self.start()
        elif not self.settings.dynamic_batching and self._worker is not None:
            await self.close()

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if not job.future.done():
                job.future.cancel()

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not self.settings.dynamic_batching:
            return await self.predictor.predict(pairs)
        if self._worker is None:
            raise RuntimeError("dynamic batcher is not started")
        future: asyncio.Future[list[float]] = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(BatchJob(pairs=pairs, future=future))
        except asyncio.QueueFull as exc:
            raise ServiceError(429, "reranker_overloaded", "reranker queue is full") from exc
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            jobs = [first]
            deadline = time.perf_counter() + self.settings.batch_window_ms / 1000
            while time.perf_counter() < deadline:
                remaining = deadline - time.perf_counter()
                try:
                    jobs.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except TimeoutError:
                    break
            await self._process(jobs)

    async def _process(self, jobs: list[BatchJob]) -> None:
        active = [job for job in jobs if not job.future.done()]
        while active:
            combined: list[tuple[str, str]] = []
            slices: list[tuple[BatchJob, int]] = []
            combined_tokens = 0
            for job in active:
                if len(combined) >= self.settings.max_batch_pairs:
                    break
                count = 0
                while job.cursor + count < len(job.pairs):
                    if len(combined) >= self.settings.max_batch_pairs:
                        break
                    pair = job.pairs[job.cursor + count]
                    pair_tokens = max(1, len(pair[0].split()) + len(pair[1].split()))
                    if combined and combined_tokens + pair_tokens > self.settings.max_batch_tokens:
                        break
                    combined.append(pair)
                    combined_tokens += pair_tokens
                    count += 1
                    if combined_tokens >= self.settings.max_batch_tokens:
                        break
                if count:
                    slices.append((job, count))
            if not combined:
                return
            try:
                scores = await self.predictor.predict(combined)
                if len(scores) != len(combined):
                    raise RuntimeError("predictor returned an invalid score count")
            except Exception as exc:
                for job in active:
                    if not job.future.done():
                        job.future.set_exception(exc)
                return
            offset = 0
            for job, count in slices:
                job.scores.extend(scores[offset : offset + count])
                job.cursor += count
                offset += count
                if job.cursor == len(job.pairs) and not job.future.done():
                    QUEUE_WAIT.observe(time.perf_counter() - job.enqueued_at)
                    job.future.set_result(job.scores)
            active = [job for job in active if not job.future.done()]
