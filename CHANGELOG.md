# Changelog

[Русский](CHANGELOG.ru.md) | **English**

All notable changes to Reranker Service are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning for
development milestones.

The repository does not currently contain release tags. Versions below reconstruct the major
milestones from the Git history; package, API, and web metadata are synchronized at `0.5.0`.

## [Unreleased]

## [0.5.0] - 2026-08-14

### Added

- Added an optional local multi-backend Compose stack with independent Jina, Alibaba, and legacy
  CrossEncoder containers behind one Nginx API proxy.
- Added static backend discovery at `GET /v1/backends` and `X-Backend` routing for the experimental
  multi-backend stack.
- Added direct local ports for backend diagnostics: `8210` for Jina, `8211` for Alibaba, and `8212`
  for the legacy runtime.
- Added `top_k` as a backward-compatible alias for `top_n` in rerank requests.
- Added documentation of the reranker's scope as a pure relevance ranking service that does not
  make business-domain decisions (match scores, requirement satisfaction, penalties).
- Added comprehensive evidence-reranking test suite: multilingual RU/EN, false-positive resilience,
  metadata preservation, top_k alias, duplicate documents, edge cases, and backward compatibility.
- Documented `score` as semantic/cross-encoder relevance (not match probability or requirement
  coverage) and `normalized_score` as optional model-specific mapping into [0, 1].

### Changed

- Made backend choice registry-driven: the UI uses the server default, retains only a valid explicit
  choice, and sends no routing header when discovery is unavailable.
- Disabled Playground text persistence by default and made browser-local storage an explicit opt-in.
- Browser-local persistence is now enabled by default for Playground and Batch Playground; users
  can opt out via the "Remember inputs in this browser" checkbox.
- Bound published Compose ports to loopback and standardized web development commands on pnpm.
- Synchronized package, API, and web version metadata at `0.5.0`.
- Expanded the web reverse-proxy timeouts for long model and reranking operations and forwarded the
  standard client/protocol headers to the API.
- Made the web package an explicit pnpm workspace and allowlisted the `esbuild` install script.
- Updated the English and Russian onboarding documentation and added this release history.
- `result.metadata` is now always populated regardless of `return_documents`, so downstream consumers
  can always track source identity even when text is not returned.

### Fixed

- Prevented the evaluation stack from loading all three reranker models into GPU memory: backend
  selection now stops every model container and starts exactly one explicit target.
- Removed Jina as the implicit UI fallback and reject unknown backend identifiers deterministically.
- Prevented the web proxy from terminating valid long-running `/v1/*` requests at the previous
  default timeout.
- Replaced the placeholder pnpm build-permission configuration with a valid workspace declaration.
- Fixed `docker-compose.multi.yml` hardcoded model environment variables that overrode `.env`
  values; now uses `${VAR:-default}` syntax so `.env` takes precedence.

## [0.4.1] - 2026-08-13

### Added

- Added expandable admin-console request details with the query, ordered documents, ranked results,
  scores, normalized scores, ranks, returned text, and cache-hit state.
- Added request timestamps to the admin request table and detail panel.
- Documented the request-history retention boundary in the API and security references.

### Changed

- Limited in-memory request-history previews to 500 characters for a query and 200 characters for
  each document.
- Kept request-history payloads in the bounded process-local deque only; they are not written to
  logs, external telemetry, or persistent server storage.

### Fixed

- Added a clear fallback in the console for legacy request records that do not contain the newer
  query, document, or result fields.
- Made the request-detail view tolerate partially populated historical records.

## [0.4.0] - 2026-08-11

### Added

- Added a backend registry with isolated ONNX pairwise, legacy CrossEncoder, Alibaba GTE pairwise,
  and Jina listwise implementations.
- Added checksum-verified, versioned ONNX artifact manifests keyed by model ID, immutable revision,
  backend, and precision.
- Added a separate model exporter that resolves revisions to full commit SHAs and never runs during
  API startup.
- Added ONNX Runtime CUDA and CPU images, plus isolated legacy, Jina, and Alibaba runtime targets.
- Added model-specific raw-score semantics and optional `normalized_score` values.
- Added provider-aware readiness metadata, CUDA validation, controlled CPU fallback, and GPU
  diagnostics.
- Added versioned browser-local persistence for the single-request Playground query, ordered
  documents, metadata, and top-N selection.
- Added document import from JSON, CSV, and plain text, plus deterministic reordering controls.
- Added reproducible latency, throughput, ranking-parity, hard-negative, image-size, and alternative
  model compatibility benchmarks.

### Changed

- Made `onnx_pairwise` with CUDA-first provider selection the default production backend.
- Kept remote-code backends opt-in and isolated behind immutable model/code revisions, explicit
  trust configuration, and exact allowlists.
- Disabled per-pair caching and cross-request batching for Jina because listwise scores depend on
  the complete candidate set.

### Fixed

- Validated ONNX artifacts with a CPU session before attempting CUDA, preventing malformed graphs
  from being misreported as CUDA initialization failures.
- Kept the active model serving when candidate validation, warm-up, or activation fails.
- Made `device=auto` degrade to a separately validated CPU session while forced CUDA continues to
  fail closed.

### Security

- Prevented the API runtime from downloading or exporting model code or weights implicitly.
- Added path confinement, symlink checks, manifest schema validation, identity checks, and SHA-256
  verification for runtime artifacts.
- Required explicit immutable allowlisting for Jina and Alibaba remote-code execution.

## [0.3.0] - 2026-08-09

### Added

- Added safe model-candidate validation, warm-up, activation, rollback, and lifecycle reporting.
- Added memory-headroom checks and restart-required reporting when two model instances cannot safely
  coexist.
- Added a reproducible PowerShell installer for Windows and CPU-only installation support.
- Added synchronized English and Russian onboarding documentation with architecture diagrams.

### Fixed

- Prevented a failed candidate load from replacing or unloading the active serving model.
- Made insufficient GPU memory an explicit lifecycle result instead of speculatively terminating
  the active runtime.

## [0.2.0] - 2026-08-08

### Added

- Added bounded cross-request dynamic micro-batching with pair, token, queue, timeout, and
  concurrency limits.
- Added runtime and cache administration endpoints, benchmark execution, aggregate metrics, health
  views, request history, and operational dashboard panels.
- Added authenticated batch reranking and expanded administrative workflows.
- Added Playground document import and stable document reordering.
- Added opt-in Redis integration coverage for TTL behavior, hashed keys, ACL failures, degraded
  inference, and recovery.

### Fixed

- Preserved request identity and result ordering when several requests share one inference batch.
- Isolated cancellation and timeout handling so one failed request cannot corrupt another request's
  result mapping.
- Made Redis failures degrade cache effectiveness without breaking readiness or inference, and
  allowed normal caching to resume after Redis recovers.

## [0.1.0] - 2026-08-08

### Added

- Added the initial production-oriented FastAPI reranking service with authenticated
  `query + document` and batch contracts.
- Added stable score ordering that preserves input order for ties.
- Added one immutable model revision and one model runtime per container.
- Added Redis score caching with SHA-256-only keys.
- Added request validation, bounded input sizes, structured errors, readiness/liveness endpoints,
  Prometheus metrics, OpenTelemetry instrumentation, and a React administration console.
- Added Docker, CPU/GPU runtime foundations, unit tests, Ruff, mypy, and coverage gates.

### Security

- Excluded bearer credentials, raw cache inputs, and normal request text from application logs.
- Added constant-time bearer comparison, non-root containers, CSP/security headers, and generic
  internal-error responses.
