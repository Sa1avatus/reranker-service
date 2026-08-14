# API

Service bearer authentication protects `/v1/rerank`, `/v1/rerank/batch`, model discovery, and `/metrics`. A distinct admin bearer protects `/v1/admin/*`. Only liveness and readiness are unauthenticated. Interactive FastAPI documentation and the HTTP OpenAPI route are disabled; the schema remains available to in-process contract tests.

Endpoints: `GET /health/live`, `GET /health/ready`, `GET /v1/models`, `GET /v1/models/current`, `POST /v1/rerank`, `POST /v1/rerank/batch`, authenticated `GET /metrics`, plus `/v1/admin/*` control-plane endpoints.

The API accepts `top_k` as a backward-compatible alias for `top_n`.  When
both are present the request is rejected as ambiguous.

`result.metadata` is always populated regardless of `return_documents`.  When
`return_documents=false` the `text` field is omitted but `metadata` is still
returned so downstream consumers can track source identity.

Admin request records (`GET /v1/admin/requests`, `GET /v1/admin/requests/{request_id}`) include the full request payload alongside technical metadata: `query` (truncated to 500 characters), `documents` (each with `id` and `text` truncated to 200 characters), and `results` (each with `id`, `score`, `normalized_score`, `rank`, `text`, `cache_hit`). Older records created before this feature was added omit these fields; the console displays a fallback message for such records.

The reranker is a pure relevance ranking service.  `score` means
**semantic/cross-encoder relevance of the document to the query** — it is not
a match probability, requirement coverage, candidate score, or any other
business-domain decision.  `normalized_score` is an optional model-specific
mapping into [0, 1].  Business logic (match calculation, requirement
satisfaction, blocker penalties, duration checks) belongs to the downstream
Matching Engine in `job-searching-assistant`.

Errors use `{"error":{"code","message","details"}}`. Codes include `invalid_request`, `request_too_large`, `validation_error`, `rate_limited`, `model_not_ready`, and `inference_timeout`. Document IDs are opaque and unchanged. Equal scores preserve input order. `truncate=false` rejects over-limit query/documents; total-character excess is always rejected.

The backward-compatible rerank response keeps `model_revision` and `device` and adds `requested_revision`, `resolved_revision`, `backend`, `rerank_mode`, and `active_provider`. Results keep `score`; `normalized_score` is optional and is only populated when a backend can justify a model-specific normalization. Pairwise backends score each query/document pair independently. `jina_listwise` scores one query against the complete ordered candidate list (up to 64 documents), so changing the candidate set may change every score. `/health/ready`, model discovery, and the admin dashboard include backend/provider metadata. A successful CPU fallback returns HTTP 200 readiness with `degraded=true` and a machine-readable reason; a missing or invalid artifact returns 503 readiness.
