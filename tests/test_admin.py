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


def test_cache_clear_confirmation(client, admin_auth):
    assert client.post("/v1/admin/cache/clear", json={}, headers=admin_auth).status_code == 400


def test_benchmark_lifecycle(client, admin_auth):
    run = client.post(
        "/v1/admin/benchmarks", json={"mode": "low_priority"}, headers=admin_auth
    ).json()
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
