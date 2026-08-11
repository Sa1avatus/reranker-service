from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

RerankPair = tuple[str, str]


@dataclass(frozen=True)
class ScoreSemantics:
    score_type: str
    score_range: tuple[float, float] | None
    higher_is_better: bool
    normalized: bool
    comparable_across_queries: bool


@dataclass(frozen=True)
class BackendCapabilities:
    rerank_mode: Literal["pairwise", "listwise"]
    supports_batching: bool
    supports_independent_scores: bool
    supports_normalized_scores: bool
    supports_cuda: bool
    supports_cpu: bool
    max_documents: int
    max_tokens: int | None
    max_length: int
    preferred_provider: str
    score_semantics: ScoreSemantics

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RerankerBackend(ABC):
    name: str
    device: str

    @abstractmethod
    async def load(self, model: str, revision: str, max_length: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def warmup(self, model: Any) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def rerank(self, model: Any, pairs: Sequence[RerankPair]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def unload(self, model: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def metadata(self, model: Any | None = None) -> dict[str, object]:
        raise NotImplementedError

    def normalized_score(self, model: Any, raw_score: float) -> float | None:
        del model, raw_score
        return None

    def candidate_requirements(self, model: str, revision: str) -> dict[str, object]:
        del model, revision
        return {}

    def reconfigure(self, model: Any, max_length: int) -> None:
        del model, max_length
