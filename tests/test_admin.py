import time


def test_admin_is_separately_protected(client, auth, admin_auth):
    assert client.get("/v1/admin/dashboard", headers=auth).status_code == 401
    assert client.get("/v1/admin/dashboard", headers=admin_auth).status_code == 200


def test_runtime_validation_and_audit(client, admin_auth):
    validated = client.post(
        "/v1/admin/runtime/validate", json={"batch_size": 32}, headers=admin_auth
    )
    assert validated.json()["valid"]
    assert (
        client.patch("/v1/admin/runtime", json={"batch_size": 32}, headers=admin_auth).status_code
        == 200
    )
    audit = client.get("/v1/admin/audit-log", headers=admin_auth).json()
    assert audit["items"][0]["action"] == "runtime.updated"
    assert client.app.state.settings.batch_size == 32


def test_runtime_patch_changes_backend_behavior_settings(client, admin_auth):
    response = client.patch(
        "/v1/admin/runtime",
        json={"dynamic_batching": False, "request_timeout_seconds": 9, "max_batch_pairs": 17},
        headers=admin_auth,
    )
    assert response.status_code == 200
    assert client.app.state.settings.dynamic_batching is False
    assert client.app.state.settings.request_timeout_seconds == 9
    assert client.app.state.settings.max_batch_pairs == 17


def test_cache_clear_confirmation(client, admin_auth):
    assert client.post("/v1/admin/cache/clear", json={}, headers=admin_auth).status_code == 400


def test_cache_patch_is_validated_and_applied(client, admin_auth):
    updated = client.patch(
        "/v1/admin/cache",
        json={"enabled": True, "ttl_seconds": 120},
        headers=admin_auth,
    )
    assert updated.status_code == 200
    assert updated.json()["ttl_seconds"] == 120
    assert client.app.state.settings.cache_enabled is True
    assert (
        client.patch("/v1/admin/cache", json={"ttl_seconds": 0}, headers=admin_auth).status_code
        == 422
    )
    assert (
        client.patch("/v1/admin/cache", json={"unknown": True}, headers=admin_auth).status_code
        == 422
    )


def test_benchmark_lifecycle(client, admin_auth):
    run = client.post(
        "/v1/admin/benchmarks",
        json={"mode": "low_priority", "repetitions": 1, "warmup_count": 0},
        headers=admin_auth,
    ).json()
    for _ in range(100):
        run = client.get(f"/v1/admin/benchmarks/{run['id']}", headers=admin_auth).json()
        if run["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert run["status"] == "completed"
    assert run["results"]["requests"] == 2
    assert run["results"]["pairs"] == 4
    assert client.post(f"/v1/admin/benchmarks/{run['id']}/baseline", headers=admin_auth).json()[
        "baseline"
    ]
    assert client.delete(f"/v1/admin/benchmarks/{run['id']}", headers=admin_auth).status_code == 200


def test_admin_operational_endpoints(client, admin_auth):
    for endpoint in ["models", "cache", "system/health", "system/resources", "requests"]:
        assert client.get(f"/v1/admin/{endpoint}", headers=admin_auth).status_code == 200
    assert client.post("/v1/admin/cache/test", headers=admin_auth).status_code == 200
    assert (
        client.post("/v1/admin/models/check", json={"name": "unknown"}, headers=admin_auth).json()[
            "allowed"
        ]
        is False
    )
    assert client.post("/v1/admin/runtime/reload", headers=admin_auth).json()["accepted"]
    assert client.post("/v1/admin/runtime/rollback", headers=admin_auth).status_code == 200


def test_admin_playground_uses_admin_token_not_service_key(client, admin_auth, payload):
    response = client.post("/v1/admin/rerank", json=payload, headers=admin_auth)
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "k8s-id"


def test_model_allowlist_and_exclusive_confirmation(client, admin_auth):
    assert (
        client.post(
            "/v1/admin/models/load", json={"name": "unknown"}, headers=admin_auth
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/admin/benchmarks", json={"mode": "exclusive"}, headers=admin_auth
        ).status_code
        == 400
    )


def test_benchmark_validation_and_cancellation(client, admin_auth):
    assert (
        client.post("/v1/admin/benchmarks", json={"repetitions": 0}, headers=admin_auth).status_code
        == 422
    )
    run = client.post("/v1/admin/benchmarks", json={"repetitions": 50}, headers=admin_auth).json()
    assert client.delete(f"/v1/admin/benchmarks/{run['id']}", headers=admin_auth).status_code == 200


def test_metrics_timeseries_contains_only_technical_metadata(client, admin_auth, auth, payload):
    assert client.post("/v1/rerank", json=payload, headers=auth).status_code == 200
    response = client.get(
        "/v1/admin/metrics/timeseries?period_seconds=3600&bucket_seconds=60",
        headers=admin_auth,
    )
    assert response.status_code == 200
    point = response.json()["points"][-1]
    assert point["requests"] >= 1
    assert point["documents"] >= 2
    assert "query" not in point
    assert "text" not in point
