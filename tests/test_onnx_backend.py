import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from reranker_service.artifacts import artifact_directory
from reranker_service.backends import onnx as onnx_module
from reranker_service.backends.onnx import CPU_PROVIDER, CUDA_PROVIDER, OnnxPairwiseBackend
from reranker_service.config import Settings

REVISION = "b" * 40


class FakeEncoding:
    ids = [1, 2, 3]
    attention_mask = [1, 1, 1]
    type_ids = [0, 0, 1]


class FakeTokenizer:
    @classmethod
    def from_file(cls, path: str) -> "FakeTokenizer":
        assert path.endswith("tokenizer.json")
        return cls()

    def enable_truncation(self, *, max_length: int) -> None:
        self.max_length = max_length

    def enable_padding(self) -> None:
        self.padding = True

    def encode_batch(
        self, pairs: list[tuple[str, str]], *, add_special_tokens: bool
    ) -> list[FakeEncoding]:
        assert add_special_tokens is True
        return [FakeEncoding() for _ in pairs]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeArray:
    def __init__(self, values: Any) -> None:
        self.values = values

    @property
    def ndim(self) -> int:
        if isinstance(self.values, list) and self.values and isinstance(self.values[0], list):
            return 2
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        if self.ndim == 2:
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def reshape(self, size: int) -> "FakeArray":
        assert size == 1
        return FakeArray([self.values])

    def __getitem__(self, item: tuple[slice, int]) -> "FakeArray":
        _, column = item
        return FakeArray([row[column] for row in self.values])

    def tolist(self) -> Any:
        return self.values


class AllFinite:
    def all(self) -> bool:
        return True


class FakeNumpy:
    int64 = object()

    @staticmethod
    def asarray(values: Any, dtype: object | None = None) -> FakeArray:
        del dtype
        return values if isinstance(values, FakeArray) else FakeArray(values)

    @staticmethod
    def isfinite(values: FakeArray) -> AllFinite:
        del values
        return AllFinite()


fake_numpy = FakeNumpy()


class FakeSession:
    def __init__(self, providers: list[str]) -> None:
        self.providers = providers

    def get_providers(self) -> list[str]:
        return self.providers

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

    def run(self, output_names: list[str] | None, inputs: dict[str, Any]) -> list[Any]:
        del output_names
        pair_count = inputs["input_ids"].shape[0]
        return [[[2.0] for _ in range(pair_count)]]


class FakeOrt:
    __version__ = "1.26.0-test"

    def __init__(self, available: list[str]) -> None:
        self.available = available
        self.fail_cpu = False
        self.fail_cuda = False
        self.created: list[list[str]] = []

    def get_available_providers(self) -> list[str]:
        return self.available

    def InferenceSession(self, model_path: str, *, providers: list[str]) -> FakeSession:
        assert model_path.endswith("model.onnx")
        self.created.append(providers)
        if providers == [CPU_PROVIDER] and self.fail_cpu:
            raise RuntimeError("invalid graph")
        if providers[0] == CUDA_PROVIDER and self.fail_cuda:
            raise RuntimeError("CUDA initialization failed")
        return FakeSession(providers)


def _settings(settings: Settings, root: Path, *, device: str = "auto") -> Settings:
    settings.backend = "onnx_pairwise"
    settings.device = device
    settings.artifact_root = root
    settings.model_revision = REVISION
    settings.precision = "fp32"
    settings.onnx_providers = f"{CUDA_PROVIDER},{CPU_PROVIDER}"
    return settings


def _write_artifact(root: Path, model_id: str) -> None:
    directory = artifact_directory(root, model_id, REVISION, "onnx_pairwise", "fp32")
    directory.mkdir(parents=True)
    contents = {"model": b"model", "tokenizer": b"tokenizer"}
    files = {"model": "model.onnx", "tokenizer": "tokenizer.json"}
    for name, relative_path in files.items():
        (directory / relative_path).write_bytes(contents[name])
    manifest = {
        "version": 1,
        "model_id": model_id,
        "requested_revision": "main",
        "resolved_revision": REVISION,
        "backend": "onnx_pairwise",
        "precision": "fp32",
        "files": files,
        "checksums": {
            name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
        },
        "onnx_opset": 17,
        "created_at": datetime.now(UTC).isoformat(),
        "supported_providers": [CUDA_PROVIDER, CPU_PROVIDER],
        "preferred_provider": CUDA_PROVIDER,
        "max_length": 512,
        "score_transform": "sigmoid",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _install_runtime(monkeypatch: pytest.MonkeyPatch, fake_ort: FakeOrt) -> None:
    monkeypatch.setattr(
        onnx_module,
        "_import_runtime_dependencies",
        lambda: (fake_numpy, fake_ort, FakeTokenizer),
    )
    monkeypatch.setattr(onnx_module, "_detect_gpu_name", lambda: "NVIDIA RTX Test")
    monkeypatch.setattr(
        onnx_module,
        "_gpu_snapshot",
        lambda: {
            "gpu_name": "NVIDIA RTX Test",
            "gpu_memory_total_bytes": 12,
            "gpu_memory_used_bytes": 6,
            "gpu_memory_free_bytes": 6,
        },
    )


def test_candidate_requirements_use_artifact_size_and_gpu_free_memory(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path)
    _write_artifact(tmp_path, configured.model)
    monkeypatch.setattr(
        onnx_module,
        "_gpu_snapshot",
        lambda: {"gpu_memory_free_bytes": 1024 * 1024 * 1024},
    )

    requirements = OnnxPairwiseBackend(configured).candidate_requirements(
        configured.model, REVISION
    )

    assert requirements["artifact_available"] is True
    assert requirements["artifact_bytes"] == len(b"model")
    assert requirements["gpu_parallel_load_supported"] is True


@pytest.mark.asyncio
async def test_auto_selects_validated_cuda_session(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path)
    _write_artifact(tmp_path, configured.model)
    fake_ort = FakeOrt([CUDA_PROVIDER, CPU_PROVIDER])
    _install_runtime(monkeypatch, fake_ort)
    backend = OnnxPairwiseBackend(configured)

    model = await backend.load(configured.model, REVISION, configured.max_length)
    scores = backend.rerank(model, [("query", "document")])

    assert model.active_provider == CUDA_PROVIDER
    assert fake_ort.created == [[CPU_PROVIDER], [CUDA_PROVIDER, CPU_PROVIDER]]
    assert scores == pytest.approx([2.0])
    assert backend.normalized_score(model, scores[0]) == pytest.approx(0.880797, rel=1e-5)
    assert backend.metadata(model)["fallback_reason"] is None
    assert backend.metadata(model)["gpu_name"] == "NVIDIA RTX Test"
    assert backend.metadata(model)["cuda_available"] is True


@pytest.mark.asyncio
async def test_auto_uses_controlled_cpu_fallback_when_cuda_is_unavailable(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path)
    _write_artifact(tmp_path, configured.model)
    fake_ort = FakeOrt([CPU_PROVIDER])
    _install_runtime(monkeypatch, fake_ort)

    model = await OnnxPairwiseBackend(configured).load(
        configured.model, REVISION, configured.max_length
    )

    assert model.active_provider == CPU_PROVIDER
    assert model.fallback_provider == CPU_PROVIDER
    assert model.fallback_reason == "cuda_unavailable_using_cpu_fallback"


@pytest.mark.asyncio
async def test_cuda_initialization_failure_falls_back_only_after_cpu_model_validation(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path)
    _write_artifact(tmp_path, configured.model)
    fake_ort = FakeOrt([CUDA_PROVIDER, CPU_PROVIDER])
    fake_ort.fail_cuda = True
    _install_runtime(monkeypatch, fake_ort)

    model = await OnnxPairwiseBackend(configured).load(
        configured.model, REVISION, configured.max_length
    )

    assert model.active_provider == CPU_PROVIDER
    assert model.fallback_reason == "cuda_initialization_failed_using_cpu_fallback"
    assert fake_ort.created[0] == [CPU_PROVIDER]


@pytest.mark.asyncio
async def test_auto_runtime_cuda_error_switches_to_validated_cpu_session(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path)
    _write_artifact(tmp_path, configured.model)
    fake_ort = FakeOrt([CUDA_PROVIDER, CPU_PROVIDER])
    _install_runtime(monkeypatch, fake_ort)
    backend = OnnxPairwiseBackend(configured)
    model = await backend.load(configured.model, REVISION, configured.max_length)

    def fail_cuda(output_names: list[str] | None, inputs: dict[str, Any]) -> list[Any]:
        del output_names, inputs
        raise RuntimeError("CUDA transient execution failure")

    model.session.run = fail_cuda
    scores = backend.rerank(model, [("query", "document")])

    assert scores == [2.0]
    assert model.active_provider == CPU_PROVIDER
    assert model.fallback_reason == "cuda_inference_failed_using_cpu_fallback"


@pytest.mark.asyncio
async def test_invalid_model_is_not_masked_as_cuda_fallback(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path)
    _write_artifact(tmp_path, configured.model)
    fake_ort = FakeOrt([CUDA_PROVIDER, CPU_PROVIDER])
    fake_ort.fail_cpu = True
    _install_runtime(monkeypatch, fake_ort)

    with pytest.raises(RuntimeError, match="invalid graph"):
        await OnnxPairwiseBackend(configured).load(
            configured.model, REVISION, configured.max_length
        )

    assert fake_ort.created == [[CPU_PROVIDER]]


@pytest.mark.asyncio
async def test_forced_cuda_fails_closed_when_provider_is_unavailable(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path, device="cuda")
    _write_artifact(tmp_path, configured.model)
    _install_runtime(monkeypatch, FakeOrt([CPU_PROVIDER]))

    with pytest.raises(RuntimeError, match="unavailable"):
        await OnnxPairwiseBackend(configured).load(
            configured.model, REVISION, configured.max_length
        )


@pytest.mark.asyncio
async def test_forced_cuda_fails_closed_when_provider_is_not_configured(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _settings(settings, tmp_path, device="cuda")
    configured.onnx_providers = CPU_PROVIDER
    _write_artifact(tmp_path, configured.model)
    _install_runtime(monkeypatch, FakeOrt([CUDA_PROVIDER, CPU_PROVIDER]))

    with pytest.raises(RuntimeError, match="not configured"):
        await OnnxPairwiseBackend(configured).load(
            configured.model, REVISION, configured.max_length
        )
