# Security

Supply service/admin secrets externally and keep them distinct. Authentication uses constant-time comparison. Tokens and document contents are excluded from logs. CORS is absent by default, generic 500 responses hide tracebacks, the API image runs non-root, cache keys hide private text, and the web proxy applies CSP and security headers.

Place TLS and distributed rate limiting at the ingress for multi-replica production. Rotate secrets, restrict health endpoints at the network boundary as appropriate, scan images/dependencies in CI, and review model artifacts before allowlisting. `/metrics` requires the service bearer token; only liveness and readiness are anonymous. In-process rate limiting is intentionally per replica.

ONNX runtime never executes Hub downloads or exporter code. The separate exporter enforces the repository allowlist, resolves the requested revision to a full 40-character commit SHA, and uses `trust_remote_code=False`. Runtime artifact paths are derived from safe encoded identities; manifests reject traversal, unknown fields or versions, identity mismatches, missing files, escaping symlinks, and checksum mismatches. Artifact manifests and provider telemetry never contain bearer tokens, authorization headers, prompts, or document text.

Remote-code backends are disabled by default. Jina v3 and Alibaba GTE require all of: an explicit
`RERANKER_TRUST_REMOTE_CODE=true`, an exact model allowlist entry, and a full immutable model SHA.
Alibaba additionally requires a full immutable SHA for its secondary code repository. These
controls limit what may execute but do not replace code and license review. The dedicated images
pin their dependency stacks, remain separate from the default ONNX runtime, and never receive API
or admin credentials during model download. Jina's CC-BY-NC-4.0 terms require deployment-specific
review. Its dedicated backend preserves true listwise inference and does not use pairwise caching
or cross-request batching.
