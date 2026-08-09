# Reranker Service

Independent FastAPI cross-encoder service and administration console. The API scores `query + document` pairs with pinned `BAAI/bge-reranker-v2-m3`, supports CPU/CUDA, Redis score caching, bounded inference concurrency, batch requests, Prometheus/OpenTelemetry, and graceful cache degradation.

## Install from scratch on Windows

Prerequisites:

- Windows 10 or 11 with PowerShell 5.1 or newer.
- Docker Desktop with the Compose v2 plugin.
- At least 8 GB of free RAM and approximately 8 GB of free disk space for the CPU image,
  model cache, and Redis data.

Clone the repository, open PowerShell in its root, and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1
```

The installer verifies Docker, creates an ignored `.env` with two independent random secrets,
builds the CPU-only image, starts Redis/API/web, and waits for the pinned model to download and
warm up. It never overwrites an existing `.env`. The first installation can take 10–30 minutes,
depending on network and CPU speed.

When readiness succeeds, open `http://localhost:8400`. Retrieve the admin token locally when you
need to sign in; do not paste it into logs or commit it:

```powershell
(Get-Content .env | Select-String '^RERANKER_ADMIN_TOKEN=').Line.Split('=', 2)[1]
```

Useful lifecycle commands:

```powershell
docker compose ps
docker compose logs -f reranker-api
docker compose down
# Preserve downloaded model and Redis data. Add -v only when you intentionally want to delete them.
```

Rerunning `.\scripts\install.ps1` is safe. Use `-SkipBuild` when the local image is already current,
or `-ReadyTimeoutMinutes 60` on a slow first download.

## Manual quick start

```powershell
Copy-Item .env.example .env
# Replace both secrets; the example values are intentionally unusable placeholders.
docker compose up -d --build
Invoke-RestMethod http://localhost:8200/health/ready
```

Open `http://localhost:8400`. For local-only development you may set `RERANKER_API_KEY=local-reranker-development-key` and `RERANKER_ADMIN_TOKEN=local-reranker-admin-token`; never deploy those values.

```powershell
$body = @{
  query = 'Kubernetes experience'
  documents = @(
    @{ id = 'a'; text = 'Ran Kubernetes in production' }
    @{ id = 'b'; text = 'SQL Server administrator' }
  )
  top_n = 2
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post -Uri http://localhost:8200/v1/rerank `
  -Headers @{ Authorization = 'Bearer local-reranker-development-key' } `
  -ContentType 'application/json' -Body $body
```

Commands: `make up`, `make down`, `make logs`, `make test`, `make lint`, `make typecheck`, `make benchmark`. One API worker is deliberate: multiple workers duplicate the model. CPU generally needs 2–6 GiB depending on model and sequence/batch sizes. CUDA additionally needs model weights, activations and allocator headroom; reduce `RERANKER_BATCH_SIZE`, `RERANKER_MAX_LENGTH`, or concurrency on OOM.

See [API.md](API.md), [OPERATIONS.md](OPERATIONS.md), and [DEVELOPMENT.md](DEVELOPMENT.md).
