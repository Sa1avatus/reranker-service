# Reranker Service

Independent FastAPI cross-encoder service and administration console. The API scores `query + document` pairs with pinned `BAAI/bge-reranker-v2-m3`, supports CPU/CUDA, Redis score caching, bounded inference concurrency, batch requests, Prometheus/OpenTelemetry, and graceful cache degradation.

## Quick start

```bash
cp .env.example .env
# Replace both secrets; the example values are intentionally unusable placeholders.
docker compose up -d --build
curl http://localhost:8200/health/ready
```

Open `http://localhost:8400`. For local-only development you may set `RERANKER_API_KEY=local-reranker-development-key` and `RERANKER_ADMIN_TOKEN=local-reranker-admin-token`; never deploy those values.

```bash
curl -s http://localhost:8200/v1/rerank \
  -H 'Authorization: Bearer local-reranker-development-key' -H 'Content-Type: application/json' \
  -d '{"query":"Kubernetes experience","documents":[{"id":"a","text":"Ran Kubernetes in production"},{"id":"b","text":"SQL Server administrator"}],"top_n":2}'
```

Commands: `make up`, `make down`, `make logs`, `make test`, `make lint`, `make typecheck`, `make benchmark`. One API worker is deliberate: multiple workers duplicate the model. CPU generally needs 2–6 GiB depending on model and sequence/batch sizes. CUDA additionally needs model weights, activations and allocator headroom; reduce `RERANKER_BATCH_SIZE`, `RERANKER_MAX_LENGTH`, or concurrency on OOM.

See [API.md](API.md), [OPERATIONS.md](OPERATIONS.md), and [DEVELOPMENT.md](DEVELOPMENT.md).

