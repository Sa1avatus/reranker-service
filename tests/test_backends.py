from collections.abc import Sequence

import pytest

from reranker_service.backends import (
    BackendCapabilities,
    BackendRegistry,
    RerankerBackend,
    ScoreSemantics,
    default_backend_registry,
)
from reranker_service.backends.alibaba import AlibabaGteBackend
from reranker_service.backends.jina import JinaListwiseBackend
from reranker_service.backends.legacy import LegacyCrossEncoderBackend
from reranker_service.runtime import ModelRuntime


class TrackingBackend(RerankerBackend):
    name = "tracking"
    device = "cpu"

    def __init__(self) -> None:
        self.loaded: list[object] = []
        self.unloaded: list[object] = []
        self.fail_rerank = False

    async def load(self, model: str, revision: str, max_length: int) -> object:
        loaded = object()
        self.loaded.append(loaded)
        return loaded

    def warmup(self, model: object) -> list[float]:
        return [0.5]

    def rerank(self, model: object, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if self.fail_rerank:
            raise RuntimeError("candidate smoke test failed")
        return [0.5] * len(pairs)

    def unload(self, model: object) -> None:
        self.unloaded.append(model)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rerank_mode="pairwise",
            supports_batching=True,
            supports_independent_scores=True,
            supports_normalized_scores=False,
            supports_cuda=False,
            supports_cpu=True,
            max_documents=100,
            max_tokens=None,
            max_length=1024,
            preferred_provider="cpu",
            score_semantics=ScoreSemantics("logit", None, True, False, False),
        )

    def metadata(self, model: object | None = None) -> dict[str, object]:
        del model
        return {"backend": self.name}


def test_default_registry_exposes_production_backends(settings):
    registry = default_backend_registry()

    assert registry.names() == (
        "alibaba_gte",
        "jina_listwise",
        "legacy_cross_encoder",
        "onnx_pairwise",
    )
    assert isinstance(registry.create("alibaba_gte", settings), AlibabaGteBackend)
    assert isinstance(registry.create("jina_listwise", settings), JinaListwiseBackend)
    assert isinstance(registry.create("legacy_cross_encoder", settings), LegacyCrossEncoderBackend)
    with pytest.raises(ValueError, match="unknown reranker backend"):
        registry.create("unknown", settings)


def test_legacy_backend_reports_pairwise_capabilities(settings):
    backend = LegacyCrossEncoderBackend(settings)

    capabilities = backend.capabilities()
    assert capabilities.rerank_mode == "pairwise"
    assert capabilities.supports_batching is True
    assert capabilities.supports_independent_scores is True
    assert capabilities.supports_cuda is True
    assert capabilities.supports_cpu is True
    assert capabilities.score_semantics.normalized is False
    assert capabilities.score_semantics.comparable_across_queries is False
    assert backend.metadata()["backend"] == "legacy_cross_encoder"


@pytest.mark.asyncio
async def test_mock_legacy_backend_loads_and_reranks_without_model_download(settings):
    backend = LegacyCrossEncoderBackend(settings)
    model = await backend.load(settings.model, settings.model_revision, settings.max_length)

    scores = backend.rerank(
        model,
        [("Python FastAPI", "Python FastAPI backend"), ("Python FastAPI", "Sous Chef")],
    )

    assert model == f"mock:{settings.model}@{settings.model_revision}"
    assert scores[0] > scores[1]
    assert backend.normalized_score(model, scores[0]) > backend.normalized_score(model, scores[1])
    assert len(backend.warmup(model)) == 1
    backend.unload(model)


def test_legacy_remote_code_requires_both_flag_and_exact_allowlist(settings):
    settings.trust_remote_code = True
    settings.remote_code_allowlist = "Alibaba-NLP/gte-multilingual-reranker-base"
    backend = LegacyCrossEncoderBackend(settings)

    assert backend.metadata()["trust_remote_code"] is False
    settings.model = "Alibaba-NLP/gte-multilingual-reranker-base"
    assert backend.metadata()["trust_remote_code"] is True


def test_registry_instances_do_not_share_mutable_factories():
    first, second = BackendRegistry(), BackendRegistry()

    first.register("legacy", LegacyCrossEncoderBackend)

    assert first.names() == ("legacy",)
    assert second.names() == ()


@pytest.mark.asyncio
async def test_runtime_uses_registry_and_unloads_failed_candidate(settings):
    backend = TrackingBackend()
    registry = BackendRegistry()
    registry.register("tracking", lambda _: backend)
    settings.backend = "tracking"
    runtime = ModelRuntime(settings, registry)
    await runtime.load()
    active_model = runtime.model
    backend.fail_rerank = True

    with pytest.raises(RuntimeError, match="candidate smoke test failed"):
        await runtime.load_candidate(settings.model, "abcdef1")

    assert runtime.model is active_model
    assert runtime.ready is True
    assert backend.unloaded == [backend.loaded[-1]]
    runtime.close()
    assert active_model in backend.unloaded


@pytest.mark.asyncio
async def test_runtime_records_failed_initial_load_operation(settings):
    class FailingWarmupBackend(TrackingBackend):
        def warmup(self, model: object) -> list[float]:
            del model
            raise RuntimeError("initial warmup failed")

    backend = FailingWarmupBackend()
    registry = BackendRegistry()
    registry.register("tracking", lambda _: backend)
    settings.backend = "tracking"
    runtime = ModelRuntime(settings, registry)

    with pytest.raises(RuntimeError, match="initial warmup failed"):
        await runtime.load()

    assert runtime.ready is False
    assert runtime.operation["type"] == "initial_load"
    assert runtime.operation["status"] == "error"
    assert runtime.operation["finished_at"] is not None
    runtime.close()
