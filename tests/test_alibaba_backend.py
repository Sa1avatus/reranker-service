import pytest

from reranker_service.backends.alibaba import AlibabaGteBackend

ALIBABA_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"
ALIBABA_REVISION = "8215cf04918ba6f7b6a62bb44238ce2953d8831c"
CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"


@pytest.mark.asyncio
async def test_mock_alibaba_backend_ranks_pairs(settings):
    settings.backend = "alibaba_gte"
    settings.model = ALIBABA_MODEL
    settings.model_revision = ALIBABA_REVISION
    backend = AlibabaGteBackend(settings)
    model = await backend.load(ALIBABA_MODEL, ALIBABA_REVISION, 512)

    scores = backend.rerank(
        model,
        [
            ("python backend", "sous chef"),
            ("python backend", "senior python backend engineer"),
        ],
    )

    assert scores[1] > scores[0]
    assert len(backend.warmup(model)) == 1
    assert backend.capabilities().rerank_mode == "pairwise"
    assert backend.normalized_score(model, -1.0) < backend.normalized_score(model, 1.0)
    backend.reconfigure(model, 9000)
    assert model.max_length == 8192


@pytest.mark.asyncio
async def test_alibaba_requires_model_and_secondary_code_sha(settings):
    settings.mock_model = False
    settings.model = ALIBABA_MODEL
    backend = AlibabaGteBackend(settings)

    with pytest.raises(RuntimeError, match="TRUST_REMOTE_CODE"):
        await backend.load(ALIBABA_MODEL, ALIBABA_REVISION, 512)
    settings.trust_remote_code = True
    with pytest.raises(RuntimeError, match="ALLOWLIST"):
        await backend.load(ALIBABA_MODEL, ALIBABA_REVISION, 512)
    settings.remote_code_allowlist = ALIBABA_MODEL
    with pytest.raises(RuntimeError, match="REMOTE_CODE_REVISION"):
        await backend.load(ALIBABA_MODEL, ALIBABA_REVISION, 512)
    settings.remote_code_revision = CODE_REVISION
    with pytest.raises(ValueError, match="only supports"):
        backend._validate_policy("other/model", ALIBABA_REVISION)


def test_alibaba_metadata_exposes_remote_code_revision(settings):
    settings.model = ALIBABA_MODEL
    settings.model_revision = ALIBABA_REVISION
    settings.remote_code_revision = CODE_REVISION
    backend = AlibabaGteBackend(settings)

    metadata = backend.metadata()

    assert metadata["resolved_revision"] == ALIBABA_REVISION
    assert metadata["remote_code_revision"] == CODE_REVISION
