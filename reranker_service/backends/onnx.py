import asyncio
import gc
import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..artifacts import ArtifactError, ArtifactStore, LoadedArtifact, artifact_directory
from ..config import Settings
from ..metrics import (
    CUDA_AVAILABLE,
    CUDA_INFERENCE_FAILURES,
    CUDA_INITIALIZATION_FAILURES,
    CUDA_OOM,
    GPU_MEMORY_FREE,
    GPU_MEMORY_USED,
)
from .base import BackendCapabilities, RerankerBackend, RerankPair, ScoreSemantics

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass
class OnnxLoadedModel:
    session: Any
    cpu_session: Any
    tokenizer: Any
    artifact: LoadedArtifact
    active_provider: str
    available_providers: tuple[str, ...]
    configured_providers: tuple[str, ...]
    fallback_provider: str | None
    fallback_reason: str | None
    runtime_version: str
    max_length: int
    gpu_name: str | None


def _import_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "onnx_pairwise requires the ONNX runtime dependencies for this image"
        ) from exc
    return numpy, onnxruntime, Tokenizer


def _gpu_snapshot() -> dict[str, object] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return None
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 4:
        return None
    try:
        total, used, free = (int(value) * 1024 * 1024 for value in values[1:])
    except ValueError:
        return None
    GPU_MEMORY_USED.set(used)
    GPU_MEMORY_FREE.set(free)
    return {
        "gpu_name": values[0][:200],
        "gpu_memory_total_bytes": total,
        "gpu_memory_used_bytes": used,
        "gpu_memory_free_bytes": free,
    }


def _detect_gpu_name() -> str | None:
    snapshot = _gpu_snapshot()
    return str(snapshot["gpu_name"]) if snapshot is not None else None


class OnnxPairwiseBackend(RerankerBackend):
    name = "onnx_pairwise"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.device
        self.store = ArtifactStore(settings.artifact_root)

    async def load(self, model: str, revision: str, max_length: int) -> OnnxLoadedModel:
        return await asyncio.to_thread(self._load_sync, model, revision, max_length)

    def _load_sync(self, model: str, revision: str, max_length: int) -> OnnxLoadedModel:
        if len(revision) != 40:
            raise ArtifactError("ONNX runtime requires a resolved 40-character revision")
        artifact = self.store.load(
            model_id=model,
            resolved_revision=revision,
            backend="onnx_pairwise",
            precision=self.settings.precision,
        )
        numpy, ort, tokenizer_type = _import_runtime_dependencies()
        configured = self.settings.configured_onnx_providers
        available = tuple(str(provider) for provider in ort.get_available_providers())
        CUDA_AVAILABLE.set(int(CUDA_PROVIDER in available))
        gpu_name = _detect_gpu_name() if CUDA_PROVIDER in available else None
        if CPU_PROVIDER not in configured or CPU_PROVIDER not in available:
            raise RuntimeError("CPUExecutionProvider is required for artifact validation")

        tokenizer = tokenizer_type.from_file(str(artifact.files["tokenizer"]))
        effective_max_length = min(max_length, artifact.manifest.max_length)
        tokenizer.enable_truncation(max_length=effective_max_length)
        tokenizer.enable_padding()

        cpu_session = ort.InferenceSession(
            str(artifact.files["model"]), providers=[CPU_PROVIDER]
        )
        cpu_model = OnnxLoadedModel(
            session=cpu_session,
            cpu_session=cpu_session,
            tokenizer=tokenizer,
            artifact=artifact,
            active_provider=CPU_PROVIDER,
            available_providers=available,
            configured_providers=configured,
            fallback_provider=None,
            fallback_reason=None,
            runtime_version=str(ort.__version__),
            max_length=effective_max_length,
            gpu_name=gpu_name,
        )
        self._infer(numpy, cpu_model, [("warmup", "warmup")])

        if self.settings.device == "cpu":
            return cpu_model
        if CUDA_PROVIDER not in configured:
            if self.settings.device == "cuda":
                raise RuntimeError("CUDAExecutionProvider is not configured")
            return cpu_model
        if CUDA_PROVIDER not in available:
            return self._fallback_or_raise(cpu_model, "cuda_unavailable_using_cpu_fallback")

        provider_order = [
            provider
            for provider in configured
            if provider in {CUDA_PROVIDER, CPU_PROVIDER} and provider in available
        ]
        try:
            cuda_session = ort.InferenceSession(
                str(artifact.files["model"]), providers=provider_order
            )
            active_providers = tuple(str(provider) for provider in cuda_session.get_providers())
            if not active_providers or active_providers[0] != CUDA_PROVIDER:
                raise RuntimeError(
                    "CUDAExecutionProvider was not selected by the inference session"
                )
            cuda_model = OnnxLoadedModel(
                session=cuda_session,
                cpu_session=cpu_session,
                tokenizer=tokenizer,
                artifact=artifact,
                active_provider=CUDA_PROVIDER,
                available_providers=available,
                configured_providers=configured,
                fallback_provider=None,
                fallback_reason=None,
                runtime_version=str(ort.__version__),
                max_length=effective_max_length,
                gpu_name=gpu_name,
            )
            self._infer(numpy, cuda_model, [("warmup", "warmup")])
            return cuda_model
        except Exception as exc:
            CUDA_INITIALIZATION_FAILURES.inc()
            if self.settings.device == "cuda" or not self.settings.cpu_fallback_enabled:
                raise RuntimeError("CUDAExecutionProvider session validation failed") from exc
            cpu_model.fallback_provider = CPU_PROVIDER
            cpu_model.fallback_reason = "cuda_initialization_failed_using_cpu_fallback"
            return cpu_model

    def _fallback_or_raise(self, cpu_model: OnnxLoadedModel, reason: str) -> OnnxLoadedModel:
        if self.settings.device == "cuda" or not self.settings.cpu_fallback_enabled:
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        cpu_model.fallback_provider = CPU_PROVIDER
        cpu_model.fallback_reason = reason
        return cpu_model

    def warmup(self, model: OnnxLoadedModel) -> list[float]:
        return self.rerank(model, [("warmup", "warmup")])

    def rerank(self, model: OnnxLoadedModel, pairs: Sequence[RerankPair]) -> list[float]:
        numpy, _, _ = _import_runtime_dependencies()
        return self._infer(numpy, model, pairs)

    def _infer(
        self, numpy: Any, model: OnnxLoadedModel, pairs: Sequence[RerankPair]
    ) -> list[float]:
        if not pairs:
            return []
        encodings = model.tokenizer.encode_batch(list(pairs), add_special_tokens=True)
        input_names = {str(item.name) for item in model.session.get_inputs()}
        supported_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        unsupported = input_names - supported_inputs
        if unsupported or "input_ids" not in input_names:
            raise RuntimeError("ONNX model has unsupported input names")
        values = {
            "input_ids": [encoding.ids for encoding in encodings],
            "attention_mask": [encoding.attention_mask for encoding in encodings],
            "token_type_ids": [encoding.type_ids for encoding in encodings],
        }
        inputs = {
            name: numpy.asarray(values[name], dtype=numpy.int64) for name in input_names
        }
        output_names = (
            [model.artifact.manifest.score_output_name]
            if model.artifact.manifest.score_output_name
            else None
        )
        try:
            outputs = model.session.run(output_names, inputs)
        except Exception as exc:
            if model.active_provider != CUDA_PROVIDER or not self._is_cuda_provider_error(exc):
                raise
            CUDA_INFERENCE_FAILURES.inc()
            is_oom = "out of memory" in str(exc).casefold()
            if is_oom:
                CUDA_OOM.inc()
            if self.settings.device == "cuda" or not self.settings.cpu_fallback_enabled:
                raise
            if not is_oom:
                try:
                    outputs = model.session.run(output_names, inputs)
                except Exception as retry_exc:
                    if not self._is_cuda_provider_error(retry_exc):
                        raise
                else:
                    return self._scores_from_outputs(numpy, outputs, len(pairs), model)
            model.session = model.cpu_session
            model.active_provider = CPU_PROVIDER
            model.fallback_provider = CPU_PROVIDER
            model.fallback_reason = "cuda_inference_failed_using_cpu_fallback"
            outputs = model.session.run(output_names, inputs)
        return self._scores_from_outputs(numpy, outputs, len(pairs), model)

    def _scores_from_outputs(
        self,
        numpy: Any,
        outputs: Sequence[Any],
        pair_count: int,
        model: OnnxLoadedModel,
    ) -> list[float]:
        if len(outputs) != 1:
            raise RuntimeError("ONNX reranker must expose exactly one selected score output")
        scores = self._extract_scores(numpy, outputs[0], pair_count, model)
        if not bool(numpy.isfinite(numpy.asarray(scores)).all()):
            raise RuntimeError("ONNX reranker returned non-finite scores")
        return scores

    @staticmethod
    def _is_cuda_provider_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return any(marker in message for marker in ("cuda", "cudnn", "cublas", "out of memory"))

    @staticmethod
    def _extract_scores(
        numpy: Any, output: Any, pair_count: int, model: OnnxLoadedModel
    ) -> list[float]:
        array = numpy.asarray(output)
        output_index = model.artifact.manifest.score_output_index
        if array.ndim == 0 and pair_count == 1:
            array = array.reshape(1)
        elif array.ndim == 2 and array.shape[0] == pair_count:
            if array.shape[1] == 1:
                array = array[:, 0]
            elif output_index is not None and output_index < array.shape[1]:
                array = array[:, output_index]
            else:
                raise RuntimeError("multi-column ONNX scores require score_output_index")
        if array.ndim != 1 or array.shape[0] != pair_count:
            raise RuntimeError("ONNX score output shape does not match the input pair count")
        return [float(value) for value in array.tolist()]

    def normalized_score(self, model: Any, raw_score: float) -> float | None:
        if not isinstance(model, OnnxLoadedModel):
            return None
        if model.artifact.manifest.score_transform != "sigmoid":
            return None
        return self._sigmoid(raw_score)

    def candidate_requirements(self, model: str, revision: str) -> dict[str, object]:
        if len(revision) != 40:
            return {"artifact_available": False}
        directory = artifact_directory(
            self.settings.artifact_root,
            model,
            revision.lower(),
            "onnx_pairwise",
            self.settings.precision,
        )
        model_path = directory / "model.onnx"
        root = self.settings.artifact_root.resolve()
        resolved_model_path = model_path.resolve()
        if not resolved_model_path.is_relative_to(root) or not resolved_model_path.is_file():
            return {"artifact_available": False}
        artifact_bytes = resolved_model_path.stat().st_size
        estimated_gpu_bytes = int(artifact_bytes * 2.5)
        snapshot = _gpu_snapshot()
        free_value = snapshot.get("gpu_memory_free_bytes") if snapshot is not None else None
        free_bytes = free_value if isinstance(free_value, int) else None
        return {
            "artifact_available": True,
            "artifact_bytes": artifact_bytes,
            "estimated_gpu_bytes": estimated_gpu_bytes,
            "gpu_memory_free_bytes": free_bytes,
            "gpu_parallel_load_supported": (
                free_bytes is not None
                and free_bytes >= estimated_gpu_bytes + 512 * 1024 * 1024
            ),
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1 / (1 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1 + exponent)

    def unload(self, model: OnnxLoadedModel) -> None:
        model.session = None
        gc.collect()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rerank_mode="pairwise",
            supports_batching=True,
            supports_independent_scores=True,
            supports_normalized_scores=True,
            supports_cuda=True,
            supports_cpu=True,
            max_documents=self.settings.max_documents,
            max_tokens=None,
            max_length=self.settings.max_length,
            preferred_provider=CUDA_PROVIDER,
            score_semantics=ScoreSemantics(
                score_type="logit",
                score_range=None,
                higher_is_better=True,
                normalized=False,
                comparable_across_queries=False,
            ),
        )

    def metadata(self, model: Any | None = None) -> dict[str, object]:
        metadata: dict[str, object] = {
            "backend": self.name,
            "requested_device": self.settings.device,
            "configured_providers": list(self.settings.configured_onnx_providers),
            "precision": self.settings.precision,
        }
        if isinstance(model, OnnxLoadedModel):
            gpu_snapshot = _gpu_snapshot() if CUDA_PROVIDER in model.available_providers else None
            metadata.update(
                {
                    "active_provider": model.active_provider,
                    "available_providers": list(model.available_providers),
                    "cuda_available": CUDA_PROVIDER in model.available_providers,
                    "gpu_name": model.gpu_name,
                    "fallback_provider": model.fallback_provider,
                    "fallback_reason": model.fallback_reason,
                    "onnxruntime_version": model.runtime_version,
                    "requested_revision": model.artifact.manifest.requested_revision,
                    "resolved_revision": model.artifact.manifest.resolved_revision,
                    "score_transform": model.artifact.manifest.score_transform,
                }
            )
            if gpu_snapshot is not None:
                metadata.update(gpu_snapshot)
        return metadata

    def reconfigure(self, model: Any, max_length: int) -> None:
        if isinstance(model, OnnxLoadedModel):
            model.max_length = min(max_length, model.artifact.manifest.max_length)
            model.tokenizer.enable_truncation(max_length=model.max_length)
