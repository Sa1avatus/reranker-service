# API

Service bearer authentication protects `/v1/rerank`, `/v1/rerank/batch`, and model discovery. A distinct admin bearer protects `/v1/admin/*`. Health and Prometheus endpoints are unauthenticated for orchestrators.

Endpoints: `GET /health/live`, `GET /health/ready`, `GET /v1/models`, `GET /v1/models/current`, `POST /v1/rerank`, `POST /v1/rerank/batch`, `GET /metrics`, plus the admin endpoints listed in OpenAPI at `/docs`.

Errors use `{"error":{"code","message","details"}}`. Codes include `invalid_request`, `request_too_large`, `validation_error`, `rate_limited`, `model_not_ready`, and `inference_timeout`. Document IDs are opaque and unchanged. Equal scores preserve input order. `truncate=false` rejects over-limit query/documents; total-character excess is always rejected.

