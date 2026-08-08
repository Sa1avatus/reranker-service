# Operations

Readiness becomes successful after model load and warm-up; liveness only reports process health. Redis failure appears as `degraded` while inference continues. Monitor `/metrics`, especially readiness, cache errors, timeouts, inference duration and in-progress requests.

CPU: use the default Docker target and `RERANKER_DEVICE=cpu`. CUDA: build `docker build --target cuda -t reranker-service:cuda .`, provide NVIDIA Container Toolkit/GPU access, and set `RERANKER_DEVICE=cuda`. CUDA selection fails fast if unavailable.

Model upgrades: add the exact model to the allowlist, pin an immutable revision, test load/warm-up/rerank in staging, benchmark against baseline, deploy canary, then promote. Roll back both name and revision. Never use a floating `main` in production.

OOM causes include long sequences, large batches, concurrent inference, two loaded models, and allocator fragmentation. Reduce max length first, then batch size/concurrency. A model transition that cannot fit two copies requires a controlled restart.

