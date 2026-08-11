# Operations

Readiness becomes successful after model load and warm-up; liveness only reports process health. Redis failure appears as `degraded` while inference continues. Monitor `/metrics`, especially readiness, cache errors, timeouts, inference duration and in-progress requests.

Redis recovery is automatic: cache operations catch connection/authentication failures and inference proceeds; subsequent operations reconnect through the Redis connection pool. The opt-in integration suite verifies real Redis TTL, hashed private keys, degraded inference during an ACL outage, and cache recovery after access is restored.

The base Compose deployment uses the ONNX GPU target. ONNX CPU uses `runtime-cpu` and `docker-compose.cpu.yml`; the legacy target is reserved for parity and compatibility validation. `docker-compose.jina.yml` and `docker-compose.alibaba.yml` opt into dedicated remote-code images and immutable revisions. GPU deployment requires Docker Desktop/WSL2 NVIDIA passthrough. The ONNX image pins ONNX Runtime 1.26.0, CUDA 12.8, and cuDNN 9. `device=auto` validates an actual CUDA session and falls back to a separately validated CPU session when allowed. Forced CUDA fails closed.

Export is a separate one-off operation through the `exporter` target and never runs during API startup. Every local artifact is addressed by allowlisted model ID, full immutable revision, backend, and precision. Startup checks the manifest identity, required files, SHA-256 checksums, tokenizer, ONNX session, and a small inference. Missing or invalid artifacts leave liveness healthy and readiness false; they are never downloaded implicitly. Readiness and the admin dashboard report the active provider, available providers, GPU name when discoverable, fallback provider, and degraded reason.

Model upgrades: add the exact model to the allowlist, pin an immutable revision, test load/warm-up/rerank in staging, benchmark against baseline, deploy canary, then promote. Roll back both name and revision. Never use a floating `main` in production.

Remote-code upgrades require a fresh code review and image rebuild. Keep the Jina and Alibaba
overrides out of a default `.env`; enable them only with their checked-in Compose override. Alibaba
must pin both `RERANKER_MODEL_REVISION` and `RERANKER_REMOTE_CODE_REVISION`. Jina is listwise, so its
cache and dynamic batching settings must remain disabled. Review model license terms before rollout.

OOM causes include long sequences, large batches, concurrent inference, two loaded models, and allocator fragmentation. The runtime reports GPU used/free memory, estimates candidate memory from the verified artifact, and refuses parallel candidate loading when headroom is insufficient. Reduce max length first, then batch size/concurrency. A model transition that cannot fit two copies is reported as requiring a controlled restart; the active model is not killed speculatively.
