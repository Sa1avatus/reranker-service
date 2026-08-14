# Architecture

`reranker-web → reranker-api → RerankService → ModelRuntime → BackendRegistry`. The web container only reverse-proxies HTTP; it has no Redis or model-cache access. FastAPI owns validation, authentication, rate limits and response contracts. `RerankService` orchestrates per-pair cache lookup, one bounded runtime and stable sorting. `ModelRuntime` owns lifecycle, provider-aware concurrency, candidate activation and rollback; the selected backend owns framework-specific load, warm-up, rerank and unload behavior. The production default is `onnx_pairwise` with CUDA-first provider selection; `legacy_cross_encoder` remains an isolated parity/rollback target. Redis is never a readiness dependency.

## Scope

The reranker is a **pure relevance ranking service**. It ranks candidate evidence
by semantic relevance to a query (atomic claim). It does not calculate match
scores, decide requirement satisfaction, apply business penalties, or produce
final match percentages. All business-domain decisions belong to the downstream
Matching Engine in `job-searching-assistant`.

`score` is a raw semantic/cross-encoder relevance value. `normalized_score` is an
optional model-specific mapping into [0, 1] (e.g. sigmoid for logit-based
backends, linear for cosine-similarity backends). Neither value represents match
probability, requirement coverage, or candidate suitability.

## Score semantics

Public `score` values are raw model logits and are not comparable across model versions. A backend may additionally return `normalized_score` only when its model-specific transform is declared. Cache keys contain only SHA-256 over immutable model revision, backend, precision, normalized texts, and relevant inference configuration. Request text is neither logged nor retained. Runtime modifications are audited; changes that affect loaded model memory are reported as restart-required.

Pairwise backends use per-pair caching and may use cross-request micro-batching. The
`jina_listwise` backend instead receives one request's complete ordered candidate set in one call;
pair caching and cross-request batching are bypassed because changing the candidate set can change
every listwise score. `alibaba_gte` and `jina_listwise` are isolated opt-in remote-code targets,
while `onnx_pairwise` remains the production default.

The lifecycle-managed micro-batcher combines pairs from concurrent requests during the configured window, caps every inference call by pair and approximate-token limits, bounds queue depth, splits oversized jobs, and maps scores back through per-request futures. Cancellation or timeout of one request does not mix request IDs or result arrays. CUDA and CPU concurrency limits are configured separately under a shared global cap.

## ONNX runtime and artifacts

The backend registry exposes `legacy_cross_encoder`, `onnx_pairwise`, `alibaba_gte`, and `jina_listwise`; API handlers and the lifecycle manager contain no framework-specific loading logic. The ONNX backend loads only local artifacts from `<root>/<encoded-model-id>/<full-sha>/onnx_pairwise/<precision>`. A strict versioned manifest binds relative files to SHA-256 checksums, the immutable model identity, exporter metadata, tokenizer metadata, provider requirements, output selection, and score transform. Paths and symlinks are resolved beneath the configured root. ONNX API startup never invokes Hub resolution, download, or export. Remote-code runtimes use separate images and require explicit immutable allowlisting.

Provider selection is candidate-local. ONNX first creates and smoke-tests a CPU session to validate graph, tokenizer, and output shape; `device=auto` then creates and smoke-tests a CUDA session. This prevents an invalid graph from being mislabeled as a CUDA failure. Only a CUDA-specific failure can activate controlled CPU fallback. Candidate sessions do not replace active provider metadata until atomic activation; failures unload the candidate and leave the active model serving.
