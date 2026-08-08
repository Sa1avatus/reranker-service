import os

import pytest
import redis.asyncio as redis

from reranker_service.cache import ScoreCache
from reranker_service.schemas import Document, RerankRequest
from reranker_service.service import RerankService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_REDIS_INTEGRATION") != "1",
        reason="set RUN_REDIS_INTEGRATION=1 with Redis available",
    ),
]


@pytest.mark.asyncio
async def test_real_redis_round_trip_ttl_and_private_key(settings):
    settings.cache_enabled = True
    settings.cache_ttl_seconds = 30
    settings.redis_url = os.getenv("REDIS_INTEGRATION_URL", "redis://127.0.0.1:57379/15")
    cache = ScoreCache(settings)
    query = "private candidate query"
    document = "private candidate document"
    try:
        await cache.client.flushdb()
        assert await cache.get_many(query, [document]) == [None]
        await cache.set_many(query, [document], [0.731])
        assert await cache.get_many(query, [document]) == [0.731]
        keys = await cache.client.keys("reranker:score:*")
        assert len(keys) == 1
        assert "private" not in keys[0]
        ttl = await cache.client.ttl(keys[0])
        assert 0 < ttl <= 30
    finally:
        await cache.client.flushdb()
        await cache.close()


@pytest.mark.asyncio
async def test_real_redis_client_recovers_after_connection_pool_disconnect(settings):
    settings.cache_enabled = True
    settings.redis_url = os.getenv("REDIS_INTEGRATION_URL", "redis://127.0.0.1:57379/15")
    cache = ScoreCache(settings)
    try:
        assert await cache.ping()
        await cache.client.connection_pool.disconnect()
        assert await cache.ping()
    finally:
        await cache.close()


class IntegrationRuntime:
    ready = True

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.75] * len(pairs)


@pytest.mark.asyncio
async def test_redis_acl_outage_degrades_cache_not_inference_and_recovers(settings):
    base_url = os.getenv("REDIS_INTEGRATION_URL", "redis://127.0.0.1:57379/15")
    admin = redis.from_url(base_url, decode_responses=True)
    username, password = "reranker_integration", "integration-only-password"
    await admin.execute_command(
        "ACL", "SETUSER", username, "reset", "on", f">{password}", "~*", "+@all"
    )
    settings.cache_enabled = True
    settings.redis_url = base_url.replace("redis://", f"redis://{username}:{password}@")
    cache = ScoreCache(settings)
    runtime = IntegrationRuntime()
    service = RerankService(settings, runtime, cache, readiness=runtime, device="cpu")
    request = RerankRequest(query="query", documents=[Document(id="doc", text="document")])
    try:
        first = await service.rerank(request)
        assert first.usage.cache_hits == 0
        await admin.execute_command("ACL", "SETUSER", username, "off")
        await cache.client.connection_pool.disconnect()
        degraded = await service.rerank(request)
        assert degraded.results[0].score == 0.75
        assert degraded.usage.cache_hits == 0
        await admin.execute_command("ACL", "SETUSER", username, "on")
        await cache.client.connection_pool.disconnect()
        await service.rerank(request)
        recovered = await service.rerank(request)
        assert recovered.usage.cache_hits == 1
    finally:
        await cache.close()
        await admin.execute_command("ACL", "DELUSER", username)
        await admin.aclose()
