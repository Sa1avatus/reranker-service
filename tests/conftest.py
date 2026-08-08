import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RERANKER_API_KEY", "test-key")
os.environ.setdefault("RERANKER_ADMIN_TOKEN", "admin-key")
os.environ.setdefault("RERANKER_MOCK_MODEL", "true")

from reranker_service.config import Settings
from reranker_service.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        RERANKER_API_KEY="test-key",
        RERANKER_ADMIN_TOKEN="admin-key",
        RERANKER_MOCK_MODEL=True,
        RERANKER_CACHE_ENABLED=False,
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings, load_model=False)) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": "Bearer test-key"}


@pytest.fixture
def admin_auth() -> dict[str, str]:
    return {"Authorization": "Bearer admin-key"}


@pytest.fixture
def payload() -> dict:
    return {
        "query": "Describe the candidate's Kubernetes experience",
        "documents": [
            {"id": "k8s-id", "text": "Kubernetes experience in production clusters."},
            {"id": "docker-id", "text": "Production experience with Docker Compose."},
        ],
        "top_n": 2,
    }
