import asyncio

import pytest

from reranker_service.batching import DynamicBatcher


class RecordingPredictor:
    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.calls: list[list[tuple[str, str]]] = []

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        if self.delay:
            await asyncio.sleep(self.delay)
        return [float(document) for _, document in pairs]


@pytest.mark.asyncio
async def test_cross_request_batching_preserves_result_isolation(settings):
    settings.dynamic_batching = True
    settings.batch_window_ms = 20
    settings.max_batch_pairs = 10
    predictor = RecordingPredictor()
    batcher = DynamicBatcher(settings, predictor)
    await batcher.start()
    try:
        first, second = await asyncio.gather(
            batcher.predict([("a", "1"), ("a", "2")]),
            batcher.predict([("b", "9")]),
        )
    finally:
        await batcher.close()
    assert first == [1.0, 2.0]
    assert second == [9.0]
    assert len(predictor.calls) == 1
    assert predictor.calls[0] == [("a", "1"), ("a", "2"), ("b", "9")]


@pytest.mark.asyncio
async def test_micro_batch_never_exceeds_pair_limit(settings):
    settings.dynamic_batching = True
    settings.batch_window_ms = 1
    settings.max_batch_pairs = 2
    predictor = RecordingPredictor()
    batcher = DynamicBatcher(settings, predictor)
    await batcher.start()
    try:
        result = await batcher.predict([("q", str(i)) for i in range(5)])
    finally:
        await batcher.close()
    assert result == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [len(call) for call in predictor.calls] == [2, 2, 1]


@pytest.mark.asyncio
async def test_cancelled_request_does_not_cancel_other_results(settings):
    settings.dynamic_batching = True
    settings.batch_window_ms = 1
    predictor = RecordingPredictor(delay=0.03)
    batcher = DynamicBatcher(settings, predictor)
    await batcher.start()
    cancelled = asyncio.create_task(batcher.predict([("cancelled", "1")]))
    survivor = asyncio.create_task(batcher.predict([("survivor", "2")]))
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(cancelled, timeout=0.01)
        assert await survivor == [2.0]
    finally:
        await batcher.close()


@pytest.mark.asyncio
async def test_disabled_batcher_calls_predictor_directly(settings):
    settings.dynamic_batching = False
    predictor = RecordingPredictor()
    batcher = DynamicBatcher(settings, predictor)
    assert await batcher.predict([("q", "3")]) == [3.0]
    assert predictor.calls == [[("q", "3")]]
