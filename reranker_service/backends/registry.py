from collections.abc import Callable

from ..config import Settings
from .alibaba import AlibabaGteBackend
from .base import RerankerBackend
from .jina import JinaListwiseBackend
from .legacy import LegacyCrossEncoderBackend
from .onnx import OnnxPairwiseBackend

BackendFactory = Callable[[Settings], RerankerBackend]


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        if not name:
            raise ValueError("backend name must not be empty")
        self._factories[name] = factory

    def create(self, name: str, settings: Settings) -> RerankerBackend:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(f"unknown reranker backend {name!r}; available: {available}") from exc
        return factory(settings)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(AlibabaGteBackend.name, AlibabaGteBackend)
    registry.register(JinaListwiseBackend.name, JinaListwiseBackend)
    registry.register(LegacyCrossEncoderBackend.name, LegacyCrossEncoderBackend)
    registry.register(OnnxPairwiseBackend.name, OnnxPairwiseBackend)
    return registry
