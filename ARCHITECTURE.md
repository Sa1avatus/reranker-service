# Architecture

`reranker-web → reranker-api → ModelRuntime / Redis`. The web container only reverse-proxies HTTP; it has no Redis or model-cache access. FastAPI owns validation, authentication, rate limits and response contracts. `RerankService` orchestrates per-pair cache lookup, one bounded runtime and stable sorting. `ModelRuntime` owns exactly one CrossEncoder and a bounded executor. Redis is never a readiness dependency.

Public scores are sigmoid-normalized logits and are not comparable across model versions. Cache keys contain only SHA-256 over the pinned model identity, normalized texts and max length. Request text is neither logged nor retained. Runtime modifications are audited; changes that affect loaded model memory are reported as restart-required.

Dynamic batching configuration and the batch-pair ceiling are exposed, but this version implements batching within each request and concurrent `/v1/rerank/batch` fan-out. It does not claim cross-request micro-batching.

