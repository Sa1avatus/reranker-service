from reranker_service import __version__


def test_liveness_and_readiness(client):
    assert client.get("/health/live").status_code == 200
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["backend"]["backend"] == "legacy_cross_encoder"
    assert response.json()["degraded"] is False


def test_application_version_is_single_source(client):
    assert client.app.version == __version__ == "0.5.0"


def test_single_backend_discovery_is_explicit_and_public(client, settings):
    response = client.get("/v1/backends")

    assert response.status_code == 200
    assert response.json() == {
        "default_backend": "legacy_cross_encoder",
        "backends": [
            {
                "id": "legacy_cross_encoder",
                "name": settings.model,
                "backend": "legacy_cross_encoder",
                "available": True,
            }
        ],
        "model_map": {settings.model: "legacy_cross_encoder"},
    }


def test_unknown_backend_header_is_rejected(client, auth):
    response = client.post(
        "/v1/rerank",
        headers={**auth, "X-Backend": "unknown"},
        json={"query": "q", "documents": [{"id": "1", "text": "d"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_backend"


def test_authentication(client, payload):
    assert client.post("/v1/rerank", json=payload).status_code == 401
    assert (
        client.post(
            "/v1/rerank", json=payload, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_rerank_order_and_ids(client, auth, payload):
    response = client.post("/v1/rerank", json=payload, headers=auth)
    assert response.status_code == 200
    results = response.json()["results"]
    assert [x["id"] for x in results] == ["k8s-id", "docker-id"]
    assert results[0]["score"] > results[1]["score"]
    assert all(0 <= x["normalized_score"] <= 1 for x in results)
    assert response.json()["requested_revision"] == "953dc6f"
    assert response.json()["resolved_revision"] == "953dc6f"
    assert response.json()["backend"] == "legacy_cross_encoder"
    assert response.json()["rerank_mode"] == "pairwise"
    assert response.json()["active_provider"] == "cpu"


def test_stable_sort(client, auth, payload):
    payload["documents"] = [
        {"id": "first", "text": "unrelated"},
        {"id": "second", "text": "unrelated"},
    ]
    results = client.post("/v1/rerank", json=payload, headers=auth).json()["results"]
    assert [x["id"] for x in results] == ["first", "second"]


def test_batch_separates_results(client, auth, payload):
    other = {**payload, "request_id": "fe88815e-e18f-4e35-86c8-f1db9428dcbb", "query": "Docker"}
    response = client.post("/v1/rerank/batch", json={"requests": [payload, other]}, headers=auth)
    assert response.status_code == 200
    assert len(response.json()["responses"]) == 2
    assert response.json()["total_pairs"] == 4


def test_validation(client, auth):
    response = client.post("/v1/rerank", json={"query": "", "documents": []}, headers=auth)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_truncation_policy(client, auth, payload, settings):
    settings.max_document_characters = 5
    payload["documents"] = [{"id": "x", "text": "too long document"}]
    payload["truncate"] = False
    response = client.post("/v1/rerank", json=payload, headers=auth)
    assert response.status_code == 413
    payload["truncate"] = True
    assert client.post("/v1/rerank", json=payload, headers=auth).json()["results"][0]["truncated"]


def test_body_size(client, auth, payload, settings):
    settings.max_body_bytes = 10
    assert client.post("/v1/rerank", json=payload, headers=auth).status_code == 413


def test_metrics(client, auth):
    assert client.get("/metrics").status_code == 401
    response = client.get("/metrics", headers=auth)
    assert response.status_code == 200
    assert "reranker_http_requests_total" in response.text


def test_openapi_contract(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    paths = client.app.openapi()["paths"]
    for path in [
        "/v1/backends",
        "/v1/rerank",
        "/v1/rerank/batch",
        "/v1/models/current",
        "/metrics",
    ]:
        assert path in paths


def test_model_revision_is_immutable_commit(settings):
    assert settings.model_revision == "953dc6f"
    assert settings.model_revision != "main"


def test_model_not_ready_error(client, auth, payload):
    client.app.state.runtime.ready = False
    response = client.post("/v1/rerank", json=payload, headers=auth)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"
