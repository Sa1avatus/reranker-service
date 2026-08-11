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
except ImportError:  # ONNX-only images intentionally omit PyTorch.
    torch = None


@dataclass
class AlibabaLoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    revision: str
    code_revision: str
    max_length: int


class AlibabaGteBackend(RerankerBackend):
    name = "alibaba_gte"
    model_id = "Alibaba-NLP/gte-multilingual-reranker-base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but unavailable for Alibaba GTE backend")
        return requested

    def _validate_policy(self, model: str, revision: str) -> None:
        if model != self.model_id:
            raise ValueError("Alibaba GTE backend only supports its pinned reranker model")
        if not self.settings.trust_remote_code:
            raise RuntimeError("Alibaba GTE backend requires RERANKER_TRUST_REMOTE_CODE=true")
        if model not in self.settings.remote_code_allowed_models:
            raise RuntimeError("Alibaba model is not in RERANKER_REMOTE_CODE_ALLOWLIST")
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise RuntimeError("Alibaba model requires an immutable 40-character revision")
        if not self.settings.remote_code_revision:
            raise RuntimeError("Alibaba model requires RERANKER_REMOTE_CODE_REVISION")

    async def load(self, model: str, revision: str, max_length: int) -> AlibabaLoadedModel:
        if self.settings.mock_model:
            return AlibabaLoadedModel(
                f"mock:{model}@{revision}", None, model, revision, "mock", max_length
            )
        self._validate_policy(model, revision)
        return await asyncio.to_thread(self._load_sync, model, revision, max_length)

    def _load_sync(self, model: str, revision: str, max_length: int) -> AlibabaLoadedModel:
        if torch is None:
            raise RuntimeError("PyTorch is required for Alibaba GTE inference")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, token=False)
        loaded = AutoModelForSequenceClassification.from_pretrained(
            model,
            revision=revision,
            code_revision=self.settings.remote_code_revision,
            trust_remote_code=True,
            token=False,
            torch_dtype=dtype,
        )
        loaded.eval()
        loaded.to(self.device)
        return AlibabaLoadedModel(
            loaded,
            tokenizer,
            model,
            revision.lower(),
            self.settings.remote_code_revision,
            max_length,
        )

    def warmup(self, model: AlibabaLoadedModel) -> list[float]:
        return self.rerank(model, [("warmup query", "warmup document")])

    def rerank(self, model: AlibabaLoadedModel, pairs: Sequence[RerankPair]) -> list[float]:
        if self.settings.mock_model:
            scores = []
            for query, document in pairs:
                query_words = set(re.findall(r"\w+", query.casefold()))
                document_words = set(re.findall(r"\w+", document.casefold()))
                scores.append(len(query_words & document_words) / max(len(query_words), 1))
            return scores
        if torch is None:
            raise RuntimeError("PyTorch is required for Alibaba GTE inference")
        inputs = model.tokenizer(
            list(pairs),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=model.max_length,
        ).to(self.device)
        with torch.inference_mode():
            raw = model.model(**inputs, return_dict=True).logits.view(-1).float().tolist()
        scores = [float(value) for value in raw]
        if len(scores) != len(pairs) or any(not math.isfinite(score) for score in scores):
            raise RuntimeError("Alibaba GTE returned invalid scores")
        return scores

    def normalized_score(self, model: Any, raw_score: float) -> float:
        del model
        if raw_score >= 0:
            return 1 / (1 + math.exp(-raw_score))
        exponent = math.exp(raw_score)
        return exponent / (1 + exponent)

    def unload(self, model: AlibabaLoadedModel) -> None:
        del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rerank_mode="pairwise",
            supports_batching=True,
            supports_independent_scores=True,
            supports_normalized_scores=True,
            supports_cuda=True,
            supports_cpu=True,
            max_documents=self.settings.max_documents,
            max_tokens=8192,
            max_length=min(self.settings.max_length, 8192),
            preferred_provider="cuda" if self.settings.device in {"auto", "cuda"} else "cpu",
            score_semantics=ScoreSemantics("logit", None, True, False, False),
        )

    def metadata(self, model: AlibabaLoadedModel | None = None) -> dict[str, object]:
        revision = (
            model.revision
            if isinstance(model, AlibabaLoadedModel)
            else self.settings.model_revision
        )
        code_revision = (
            model.code_revision
            if isinstance(model, AlibabaLoadedModel)
            else self.settings.remote_code_revision
        )
        available = ["cpu"]
        if torch is not None and torch.cuda.is_available():
            available.insert(0, "cuda")
        return {
            "backend": self.name,
            "requested_revision": revision,
            "resolved_revision": revision,
            "remote_code_revision": code_revision,
            "requested_device": self.settings.device,
            "active_provider": self.device,
            "available_providers": available,
            "precision": "fp16" if self.device == "cuda" else "fp32",
            "trust_remote_code": self.settings.trust_remote_code,
        }

    def reconfigure(self, model: AlibabaLoadedModel, max_length: int) -> None:
        model.max_length = min(max_length, 8192)
