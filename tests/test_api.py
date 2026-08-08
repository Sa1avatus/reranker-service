def test_liveness_and_readiness(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


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
    assert all(0 <= x["score"] <= 1 for x in results)


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


def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "reranker_http_requests_total" in response.text


def test_openapi_contract(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ["/v1/rerank", "/v1/rerank/batch", "/v1/models/current", "/metrics"]:
        assert path in paths


def test_model_revision_is_immutable_commit(settings):
    assert settings.model_revision == "953dc6f"
    assert settings.model_revision != "main"


def test_model_not_ready_error(client, auth, payload):
    client.app.state.runtime.ready = False
    response = client.post("/v1/rerank", json=payload, headers=auth)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"
