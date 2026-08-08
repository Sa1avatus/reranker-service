import time
from collections import deque
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
