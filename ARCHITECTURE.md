# Architecture

`reranker-web → reranker-api → ModelRuntime / Redis`. The web container only reverse-proxies HTTP; it has no Redis or model-cache access. FastAPI owns validation, authentication, rate limits and response contracts. `RerankService` orchestrates per-pair cache lookup, one bounded runtime and stable sorting. `ModelRuntime` owns exactly one CrossEncoder and a bounded executor. Redis is never a readiness dependency.

Public scores are sigmoid-normalized logits and are not comparable across model versions. Cache keys contain only SHA-256 over the pinned model identity, normalized texts and max length. Request text is neither logged nor retained. Runtime modifications are audited; changes that affect loaded model memory are reported as restart-required.

The lifecycle-managed micro-batcher combines pairs from concurrent requests during the configured window, caps every inference call at `RERANKER_MAX_BATCH_PAIRS`, splits oversized jobs, and maps scores back through per-request futures. Cancellation or timeout of one request does not mix request IDs or result arrays.
