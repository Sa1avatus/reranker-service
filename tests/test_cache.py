import pytest

from reranker_service.cache import ScoreCache


def test_cache_key_is_private_and_stable(settings):
    cache = ScoreCache(settings)
    key = cache.key("private query", "private document")
    assert key == cache.key(" private   query ", "private document")
    assert "private" not in key


@pytest.mark.asyncio
async def test_redis_failure_degrades(settings):
    settings.cache_enabled = True
    settings.redis_url = "redis://127.0.0.1:1/0"
    cache = ScoreCache(settings)
    assert await cache.get_many("q", ["d"]) == [None]
    await cache.set_many("q", ["d"], [0.5])
    assert not await cache.ping()
    await cache.close()
