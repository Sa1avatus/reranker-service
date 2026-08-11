import pytest

from reranker_service.cache import ScoreCache


def test_cache_key_is_private_and_stable(settings):
    cache = ScoreCache(settings)
    key = cache.key("private query", "private document")
    assert key == cache.key(" private   query ", "private document")
    assert "private" not in key


def test_cache_key_separates_backend_precision_and_revision(settings):
    cache = ScoreCache(settings)
    original = cache.key("query", "document")

    settings.backend = "onnx_pairwise"
    assert cache.key("query", "document") != original
    onnx_fp32 = cache.key("query", "document")

    settings.precision = "fp16"
    assert cache.key("query", "document") != onnx_fp32
    onnx_fp16 = cache.key("query", "document")

    settings.model_revision = "f" * 40
    assert cache.key("query", "document") != onnx_fp16


@pytest.mark.asyncio
async def test_redis_failure_degrades(settings):
    settings.cache_enabled = True
    settings.redis_url = "redis://127.0.0.1:1/0"
    cache = ScoreCache(settings)
    assert await cache.get_many("q", ["d"]) == [None]
    await cache.set_many("q", ["d"], [0.5])
    assert not await cache.ping()
    await cache.close()
