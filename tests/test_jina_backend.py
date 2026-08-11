import pytest
from fastapi.testclient import TestClient

from reranker_service.backends.jina import JinaListwiseBackend, JinaLoadedModel
from reranker_service.main import create_app

JINA_MODEL = "jinaai/jina-reranker-v3"
JINA_REVISION = "d7d7e73b6ea138ced340b83865931b5dfb6c97aa"


@pytest.mark.asyncio
async def test_jina_requires_explicit_remote_code_policy(settings):
    settings.mock_model = False
    settings.model = JINA_MODEL
    settings.model_revision = JINA_REVISION
    backend = JinaListwiseBackend(settings)

    with pytest.raises(RuntimeError, match="TRUST_REMOTE_CODE"):
        await backend.load(JINA_MODEL, JINA_REVISION, 1024)

    settings.trust_remote_code = True
    with pytest.raises(RuntimeError, match="REMOTE_CODE_ALLOWLIST"):
        await backend.load(JINA_MODEL, JINA_REVISION, 1024)


@pytest.mark.asyncio
async def test_mock_jina_is_listwise_and_preserves_input_score_order(settings):
    settings.backend = "jina_listwise"
    settings.model = JINA_MODEL
    settings.model_revision = JINA_REVISION
    backend = JinaListwiseBackend(settings)
    model = await backend.load(JINA_MODEL, JINA_REVISION, 1024)

    scores = backend.rerank(
        model,
        [
            ("python backend", "sous chef"),
            ("python backend", "senior python backend engineer"),
        ],
    )

    assert scores[1] > scores[0]
    assert len(backend.warmup(model)) == 1
    assert backend.capabilities().rerank_mode == "listwise"
    assert backend.capabilities().supports_independent_scores is False
    with pytest.raises(ValueError, match="different queries"):
        backend.rerank(model, [("first", "doc"), ("second", "doc")])


def test_jina_maps_sorted_results_back_to_input_order(settings):
    class FakeModel:
        def rerank(self, *args, **kwargs):
            del args, kwargs
            return [
                {"index": 1, "relevance_score": 0.75},
                {"index": 0, "relevance_score": -0.25},
            ]

    settings.mock_model = False
    settings.backend = "jina_listwise"
    backend = JinaListwiseBackend(settings)
    loaded = JinaLoadedModel(FakeModel(), JINA_MODEL, JINA_REVISION, 1024)

    scores = backend.rerank(loaded, [("query", "first"), ("query", "second")])

    assert scores == [-0.25, 0.75]
    assert backend.normalized_score(loaded, -0.25) == 0.375
    assert backend.normalized_score(loaded, 2.0) is None
    assert backend.metadata(loaded)["resolved_revision"] == JINA_REVISION
    assert backend.candidate_requirements(JINA_MODEL, JINA_REVISION)["artifact_available"]
    backend.reconfigure(loaded, 512)
    assert loaded.max_length == 512


def test_jina_remote_code_policy_rejects_wrong_model_and_mutable_revision(settings):
    settings.trust_remote_code = True
    settings.remote_code_allowlist = JINA_MODEL
    backend = JinaListwiseBackend(settings)

    with pytest.raises(ValueError, match="only supports"):
        backend._validate_remote_code_policy("other/model", JINA_REVISION)
    with pytest.raises(RuntimeError, match="immutable"):
        backend._validate_remote_code_policy(JINA_MODEL, "main")


def test_listwise_api_bypasses_pairwise_cache_and_batching(settings):
    settings.backend = "jina_listwise"
    settings.model = JINA_MODEL
    settings.model_revision = JINA_REVISION
    settings.model_allowlist = JINA_MODEL
    settings.cache_enabled = True
    settings.redis_url = "redis://127.0.0.1:1/0"
    with TestClient(create_app(settings, load_model=False)) as client:
        response = client.post(
            "/v1/rerank",
            headers={"Authorization": "Bearer test-key"},
            json={
                "query": "python backend",
                "documents": [
                    {"id": "negative", "text": "sous chef"},
                    {"id": "relevant", "text": "senior python backend engineer"},
                ],
                "top_n": 2,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "jina_listwise"
    assert payload["rerank_mode"] == "listwise"
    assert payload["usage"]["cache_hits"] == 0
    assert [result["id"] for result in payload["results"]] == ["relevant", "negative"]
