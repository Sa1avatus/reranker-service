import hashlib
import json
import unicodedata
from collections.abc import Sequence

import redis.asyncio as redis

from .config import Settings
from .metrics import CACHE_ERRORS, CACHE_HITS, CACHE_MISSES


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


class ScoreCache:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[no-untyped-call]

    def key(self, query: str, document: str) -> str:
        payload = "\0".join(
            (
                self.settings.model,
                self.settings.model_revision,
                self.settings.backend,
                self.settings.precision,
                normalize_text(query),
                normalize_text(document),
                str(self.settings.max_length),
            )
        )
        return "reranker:score:" + hashlib.sha256(payload.encode()).hexdigest()

    async def get_many(self, query: str, documents: Sequence[str]) -> list[float | None]:
        if not self.settings.cache_enabled:
            return [None] * len(documents)
        try:
            values = await self.client.mget([self.key(query, d) for d in documents])
            result = [float(v) if v is not None else None for v in values]
            CACHE_HITS.inc(sum(v is not None for v in result))
            CACHE_MISSES.inc(sum(v is None for v in result))
            return result
        except Exception:
            CACHE_ERRORS.inc()
            return [None] * len(documents)

    async def set_many(self, query: str, documents: Sequence[str], scores: Sequence[float]) -> None:
        if not self.settings.cache_enabled or not documents:
            return
        try:
            async with self.client.pipeline(transaction=False) as pipe:
                for doc, score in zip(documents, scores, strict=True):
                    pipe.set(
                        self.key(query, doc), json.dumps(score), ex=self.settings.cache_ttl_seconds
                    )
                await pipe.execute()
        except Exception:
            CACHE_ERRORS.inc()

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def clear(self, model_only: bool = False) -> int:
        prefix = "reranker:score:*"
        count = 0
        async for key in self.client.scan_iter(match=prefix, count=500):
            count += await self.client.delete(key)
        return count

    async def close(self) -> None:
        await self.client.aclose()
