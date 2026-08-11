import asyncio
import gc
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .base import BackendCapabilities, RerankerBackend, RerankPair, ScoreSemantics

try:
    import torch
except ImportError:  # Unit-test and ONNX-only images intentionally omit PyTorch.
    torch = None


@dataclass
class JinaLoadedModel:
    model: Any
    model_id: str
    revision: str
    max_length: int


class JinaListwiseBackend(RerankerBackend):
    name = "jina_listwise"
    model_id = "jinaai/jina-reranker-v3"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but unavailable for Jina listwise backend")
        return requested

    def _validate_remote_code_policy(self, model: str, revision: str) -> None:
        if model != self.model_id:
            raise ValueError("Jina listwise backend only supports jinaai/jina-reranker-v3")
        if not self.settings.trust_remote_code:
            raise RuntimeError("Jina listwise backend requires RERANKER_TRUST_REMOTE_CODE=true")
        if model not in self.settings.remote_code_allowed_models:
            raise RuntimeError("Jina model is not in RERANKER_REMOTE_CODE_ALLOWLIST")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise RuntimeError("Jina remote code requires an immutable 40-character revision")

    async def load(self, model: str, revision: str, max_length: int) -> JinaLoadedModel:
        if self.settings.mock_model:
            return JinaLoadedModel(f"mock:{model}@{revision}", model, revision, max_length)
        self._validate_remote_code_policy(model, revision)
        return await asyncio.to_thread(self._load_sync, model, revision, max_length)

    def _load_sync(self, model: str, revision: str, max_length: int) -> JinaLoadedModel:
        if torch is None:
            raise RuntimeError("PyTorch is required for Jina listwise inference")
        from transformers import AutoModel, AutoTokenizer

        dtype = (
            torch.float16
            if self.device == "cuda" and self.settings.precision == "fp16"
            else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=True,
            token=False,
        )
        loaded = AutoModel.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=True,
            token=False,
            dtype=dtype,
        )
        loaded._tokenizer = tokenizer
        loaded.eval()
        loaded.to(self.device)
        return JinaLoadedModel(loaded, model, revision.lower(), max_length)

    def warmup(self, model: JinaLoadedModel) -> list[float]:
        return self.rerank(model, [("warmup query", "warmup document")])

    def rerank(self, model: JinaLoadedModel, pairs: Sequence[RerankPair]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        if any(pair_query != query for pair_query, _ in pairs):
            raise ValueError("listwise inference cannot combine different queries")
        documents = [document for _, document in pairs]
        if len(documents) > self.capabilities().max_documents:
            raise ValueError("too many documents for Jina listwise inference")
        if self.settings.mock_model:
            query_words = set(re.findall(r"\w+", query.casefold()))
            return [
                len(query_words & set(re.findall(r"\w+", document.casefold())))
                / max(len(query_words), 1)
                for document in documents
            ]
        max_doc_length = min(max(model.max_length, 64), 8192)
        ranked = model.model.rerank(
            query,
            documents,
            top_n=None,
            return_embeddings=False,
            max_doc_length=max_doc_length,
            max_query_length=min(512, max_doc_length),
        )
        scores: list[float | None] = [None] * len(documents)
        for result in ranked:
            index = int(result["index"])
            if index < 0 or index >= len(scores) or scores[index] is not None:
                raise RuntimeError("Jina returned invalid document indexes")
            score = float(result["relevance_score"])
            if not math.isfinite(score):
                raise RuntimeError("Jina returned a non-finite relevance score")
            scores[index] = score
        if any(score is None for score in scores):
            raise RuntimeError("Jina did not return every document")
        return [float(score) for score in scores if score is not None]

    def normalized_score(self, model: Any, raw_score: float) -> float | None:
        del model
        if not math.isfinite(raw_score) or not -1 <= raw_score <= 1:
            return None
        return (raw_score + 1.0) / 2.0

    def unload(self, model: JinaLoadedModel) -> None:
        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rerank_mode="listwise",
            supports_batching=False,
            supports_independent_scores=False,
            supports_normalized_scores=True,
            supports_cuda=True,
            supports_cpu=True,
            max_documents=min(self.settings.max_documents, 64),
            max_tokens=131072,
            max_length=min(self.settings.max_length, 8192),
            preferred_provider="cuda" if self.settings.device in {"auto", "cuda"} else "cpu",
            score_semantics=ScoreSemantics(
                score_type="cosine_similarity",
                score_range=(-1.0, 1.0),
                higher_is_better=True,
                normalized=False,
                comparable_across_queries=False,
            ),
        )

    def metadata(self, model: JinaLoadedModel | None = None) -> dict[str, object]:
        available = ["cpu"]
        if torch is not None and torch.cuda.is_available():
            available.insert(0, "cuda")
        revision = (
            model.revision if isinstance(model, JinaLoadedModel) else self.settings.model_revision
        )
        return {
            "backend": self.name,
            "requested_revision": revision,
            "resolved_revision": revision,
            "requested_device": self.settings.device,
            "active_provider": self.device,
            "available_providers": available,
            "precision": self.settings.precision if self.device == "cuda" else "fp32",
            "trust_remote_code": self.settings.trust_remote_code,
        }

    def candidate_requirements(self, model: str, revision: str) -> dict[str, object]:
        del model, revision
        estimated_bytes = 3_500_000_000
        free_bytes: int | None = None
        if torch is not None and torch.cuda.is_available():
            free_bytes = int(torch.cuda.mem_get_info()[0])
        return {
            "artifact_available": True,
            "estimated_gpu_bytes": estimated_bytes,
            "gpu_free_bytes": free_bytes,
            "gpu_parallel_load_supported": free_bytes is None or free_bytes >= estimated_bytes,
        }

    def reconfigure(self, model: JinaLoadedModel, max_length: int) -> None:
        model.max_length = max_length
