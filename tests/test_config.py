import pytest

from reranker_service.config import Settings


def test_onnx_backend_environment_alias_is_normalized() -> None:
    settings = Settings(
        RERANKER_API_KEY="test-key",
        RERANKER_ADMIN_TOKEN="admin-key",
        RERANKER_BACKEND="onnx",
    )

    assert settings.backend == "onnx_pairwise"


def test_remote_code_revision_must_be_immutable() -> None:
    with pytest.raises(ValueError, match="immutable 40-character SHA"):
        Settings(
            RERANKER_API_KEY="test-key",
            RERANKER_ADMIN_TOKEN="admin-key",
            RERANKER_REMOTE_CODE_REVISION="main",
        )


@pytest.mark.parametrize(
    ("providers", "message"),
    [
        ("CPUExecutionProvider,CPUExecutionProvider", "without duplicates"),
        ("UnknownExecutionProvider", "unsupported ONNX execution provider"),
    ],
)
def test_onnx_provider_list_is_strict(providers: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(
            RERANKER_API_KEY="test-key",
            RERANKER_ADMIN_TOKEN="admin-key",
            RERANKER_ONNX_PROVIDERS=providers,
        )
