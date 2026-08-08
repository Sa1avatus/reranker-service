import asyncio
import statistics
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any
from uuid import uuid4

import psutil

from .schemas import BenchmarkSpec, Document, RerankRequest
from .service import RerankService

DATASET = (
    (
        "Describe the candidate's Kubernetes experience",
        (
            "Basic conceptual knowledge of Kubernetes. No hands-on production experience.",
            "Production experience with Docker and Docker Compose.",
            "Strong experience administering Microsoft SQL Server.",
        ),
    ),
    (
        "Опишите опыт кандидата с Python",
        (
            "Пять лет разрабатывал backend-сервисы на Python и FastAPI.",
            "Администрировал кластеры Microsoft SQL Server.",
            "Работал с Docker и Kubernetes в production.",
        ),
    ),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


class BenchmarkRunner:
    def __init__(
        self,
        service: RerankService,
        runs: dict[str, dict[str, Any]],
        audit: Callable[[str, dict[str, Any] | None], None],
    ) -> None:
        self.service = service
        self.runs = runs
        self.audit = audit
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def start(self, spec: BenchmarkSpec) -> dict[str, Any]:
        benchmark_id = str(uuid4())
        run: dict[str, Any] = {
            "id": benchmark_id,
            "status": "queued",
            "created_at": time.time(),
            "parameters": spec.model_dump(exclude={"confirm"}),
            "baseline": False,
            "results": None,
            "error": None,
        }
        self.runs[benchmark_id] = run
        self.tasks[benchmark_id] = asyncio.create_task(
            self._execute(benchmark_id, spec), name=f"benchmark-{benchmark_id}"
        )
        self.audit("benchmark.created", {"id": benchmark_id, "mode": spec.mode})
        return run

    async def delete(self, benchmark_id: str) -> bool:
        task = self.tasks.pop(benchmark_id, None)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        existed = self.runs.pop(benchmark_id, None) is not None
        if existed:
            self.audit("benchmark.deleted", {"id": benchmark_id})
        return existed

    async def close(self) -> None:
        for benchmark_id in list(self.tasks):
            await self.delete(benchmark_id)

    async def _execute(self, benchmark_id: str, spec: BenchmarkSpec) -> None:
        run = self.runs[benchmark_id]
        run["status"] = "running"
        run["started_at"] = time.time()
        latencies: list[float] = []
        pairs = 0
        process = psutil.Process()
        memory_before = process.memory_info().rss
        try:
            cases = DATASET if spec.multilingual else DATASET[:1]
            for iteration in range(spec.warmup_count + spec.repetitions):
                for query, candidates in cases:
                    selected = [
                        candidates[index % len(candidates)] for index in range(spec.document_count)
                    ]
                    request = RerankRequest(
                        query=query,
                        documents=[
                            Document(id=f"bench-{index}", text=text)
                            for index, text in enumerate(selected)
                        ],
                        top_n=spec.document_count,
                        return_documents=False,
                    )
                    started = time.perf_counter()
                    await self.service.rerank(request)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if iteration >= spec.warmup_count:
                        latencies.append(elapsed_ms)
                        pairs += len(selected)
                    if spec.mode == "low_priority":
                        await asyncio.sleep(0.01)
            seconds = sum(latencies) / 1000
            run["results"] = {
                "requests": len(latencies),
                "pairs": pairs,
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "p99_ms": percentile(latencies, 0.99),
                "mean_ms": statistics.mean(latencies),
                "pairs_per_second": pairs / seconds if seconds else 0,
                "documents_per_second": pairs / seconds if seconds else 0,
                "process_memory_before_bytes": memory_before,
                "process_memory_after_bytes": process.memory_info().rss,
            }
            run["status"] = "completed"
            self.audit("benchmark.completed", {"id": benchmark_id})
        except asyncio.CancelledError:
            run["status"] = "cancelled"
            raise
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = type(exc).__name__
            self.audit("benchmark.failed", {"id": benchmark_id, "error": type(exc).__name__})
        finally:
            run["finished_at"] = time.time()
