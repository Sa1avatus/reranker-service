param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("legacy", "jina", "alibaba")]
    [string]$Backend
)

$ErrorActionPreference = "Stop"
$composeFile = Join-Path $PSScriptRoot "..\docker-compose.multi.yml"
$definitions = @{
    legacy = @{
        Service = "reranker-api-legacy"
        Upstream = "reranker-api-legacy:8200"
        Model = "BAAI/bge-reranker-v2-m3"
        Kind = "legacy_cross_encoder"
    }
    jina = @{
        Service = "reranker-api-jina"
        Upstream = "reranker-api-jina:8200"
        Model = "jinaai/jina-reranker-v3"
        Kind = "jina_listwise"
    }
    alibaba = @{
        Service = "reranker-api-alibaba"
        Upstream = "reranker-api-alibaba:8200"
        Model = "Alibaba-NLP/gte-multilingual-reranker-base"
        Kind = "alibaba_gte"
    }
}
$selected = $definitions[$Backend]

# Stop every model process first so a previous selection cannot retain GPU memory.
docker compose -f $composeFile stop reranker-api-jina reranker-api-alibaba reranker-api-legacy
if ($LASTEXITCODE -ne 0) { throw "Failed to stop existing reranker backends" }

$env:RERANKER_DEFAULT_BACKEND = $Backend
$env:RERANKER_UPSTREAM = $selected.Upstream
$env:RERANKER_MODEL_NAME = $selected.Model
$env:RERANKER_BACKEND_KIND = $selected.Kind

docker compose -f $composeFile --profile $Backend up --build -d `
    reranker-redis $selected.Service proxy reranker-web
if ($LASTEXITCODE -ne 0) { throw "Failed to start the selected reranker backend" }

Write-Output "Selected backend: $Backend ($($selected.Model))"
Write-Output "Only service $($selected.Service) is running a model; other backend containers are stopped."
