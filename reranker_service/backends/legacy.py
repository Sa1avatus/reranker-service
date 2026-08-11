import asyncio
import math
import re
from collections.abc import Sequence
from typing import Any

from ..config import Settings
from .base import BackendCapabilities, RerankerBackend, RerankPair, ScoreSemantics

try:
    import torch
except ImportError:  # Unit-test/minimal control-plane installations may omit the model runtime.
    torch = None


class LegacyCrossEncoderBackend(RerankerBackend):
    name = "legacy_cross_encoder"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but unavailable")
        return requested

    async def load(self, model: str, revision: str, max_length: int) -> Any:
        if self.settings.mock_model:
            return f"mock:{model}@{revision}"
        from sentence_transformers import CrossEncoder

        trust_remote_code = (
            self.settings.trust_remote_code and model in self.settings.remote_code_allowed_models
        )

        return await asyncio.to_thread(
            CrossEncoder,
            model,
            revision=revision,
            device=self.device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

    def _mock_rerank(self, pairs: Sequence[RerankPair]) -> list[float]:
        scores = []
        for query, document in pairs:
            query_words = set(re.findall(r"\w+", query.casefold()))
            document_words = set(re.findall(r"\w+", document.casefold()))
            overlap = len(query_words & document_words) / max(len(query_words), 1)
            scores.append(overlap * 8 - 3)
        return scores

    def warmup(self, model: Any) -> list[float]:
        return self.rerank(model, [("warmup", "warmup")])

    def rerank(self, model: Any, pairs: Sequence[RerankPair]) -> list[float]:
        if self.settings.mock_model:
            return self._mock_rerank(pairs)
        if torch is None:
            raise RuntimeError("PyTorch is required for legacy inference")
        with torch.inference_mode():
            kwargs = {
                "batch_size": self.settings.batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "activation_fn": torch.nn.Identity(),
            }
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw = model.predict(list(pairs), **kwargs)
            else:
                raw = model.predict(list(pairs), **kwargs)
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [float(value) for value in values]

    def normalized_score(self, model: Any, raw_score: float) -> float:
        del model
        if raw_score >= 0:
            return 1 / (1 + math.exp(-raw_score))
        exponent = math.exp(raw_score)
        return exponent / (1 + exponent)

    def unload(self, model: Any) -> None:
        del model
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
            max_tokens=None,
            max_length=self.settings.max_length,
            preferred_provider="cuda" if self.settings.device in {"auto", "cuda"} else "cpu",
            score_semantics=ScoreSemantics(
                score_type="logit",
                score_range=None,
                higher_is_better=True,
                normalized=False,
                comparable_across_queries=False,
            ),
        )

    def metadata(self, model: Any | None = None) -> dict[str, object]:
        del model
        available_providers = ["cpu"]
        if torch is not None and torch.cuda.is_available():
            available_providers.insert(0, "cuda")
        return {
            "backend": self.name,
            "requested_device": self.settings.device,
            "active_provider": self.device,
            "available_providers": available_providers,
            "precision": "fp16" if self.device == "cuda" else "fp32",
            "trust_remote_code": (
                self.settings.trust_remote_code
                and self.settings.model in self.settings.remote_code_allowed_models
            ),
        }

    def reconfigure(self, model: Any, max_length: int) -> None:
        if not self.settings.mock_model:
            model.max_length = max_length
