import time
from collections import defaultdict, deque
from typing import Any
from uuid import uuid4

import psutil

try:
    import torch
except ImportError:
    torch = None

from .config import Settings
from .schemas import RuntimePatch


class AdminState:
    def __init__(self, settings: Settings) -> None:
        self.started = time.time()
        self.runtime = {name: getattr(settings, name) for name in RuntimePatch.model_fields}
        self.previous = dict(self.runtime)
        self.audit: deque[dict[str, Any]] = deque(maxlen=1000)
        self.requests: deque[dict[str, Any]] = deque(maxlen=1000)
        self.benchmarks: dict[str, dict[str, Any]] = {}

    def record(self, action: str, details: dict[str, Any] | None = None) -> None:
        self.audit.appendleft(
            {
                "id": str(uuid4()),
                "timestamp": time.time(),
                "action": action,
                "details": details or {},
            }
        )

    def validate(self, patch: RuntimePatch) -> dict[str, Any]:
        candidate = self.runtime | patch.model_dump(exclude_none=True)
        memory_warning = candidate["batch_size"] * candidate["max_length"] > 32768
        return {
            "valid": True,
            "configuration": candidate,
            "restart_required": False,
            "memory_warning": memory_warning,
        }

    def apply(self, patch: RuntimePatch) -> dict[str, Any]:
        result = self.validate(patch)
        self.previous, self.runtime = dict(self.runtime), result["configuration"]
        self.record("runtime.updated", patch.model_dump(exclude_none=True))
        return result

    def resources(self) -> dict[str, Any]:
        process = psutil.Process()
        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "process_memory_bytes": process.memory_info().rss,
            "cuda_available": torch is not None and torch.cuda.is_available(),
            "gpu_memory_bytes": torch.cuda.memory_allocated()
            if torch is not None and torch.cuda.is_available()
            else None,
            "uptime_seconds": round(time.time() - self.started),
        }

    def timeseries(self, period_seconds: int, bucket_seconds: int) -> list[dict[str, Any]]:
        cutoff = time.time() - period_seconds
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for request in self.requests:
            timestamp = float(request["timestamp"])
            if timestamp >= cutoff:
                bucket = int(timestamp // bucket_seconds) * bucket_seconds
                buckets[bucket].append(request)
        points = []
        for timestamp, requests in sorted(buckets.items()):
            latencies = sorted(float(item["latency_ms"]) for item in requests)
            p95_index = min(round((len(latencies) - 1) * 0.95), len(latencies) - 1)
            points.append(
                {
                    "timestamp": timestamp,
                    "requests": len(requests),
                    "documents": sum(int(item["documents_count"]) for item in requests),
                    "cache_hits": sum(int(item["cache_hits"]) for item in requests),
                    "latency_mean_ms": sum(latencies) / len(latencies),
                    "latency_p95_ms": latencies[p95_index],
                    "errors": sum(item["status"] != "success" for item in requests),
                }
            )
        return points
