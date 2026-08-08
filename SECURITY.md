# Security

Supply service/admin secrets externally and keep them distinct. Authentication uses constant-time comparison. Tokens and document contents are excluded from logs. CORS is absent by default, generic 500 responses hide tracebacks, the API image runs non-root, cache keys hide private text, and the web proxy applies CSP and security headers.

Place TLS and distributed rate limiting at the ingress for multi-replica production. Rotate secrets, restrict `/metrics` and health endpoints at the network boundary as appropriate, scan images/dependencies in CI, and review model artifacts before allowlisting. In-process rate limiting is intentionally per replica.

