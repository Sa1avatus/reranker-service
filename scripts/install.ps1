[CmdletBinding()]
param(
    [int]$ReadyTimeoutMinutes = 30,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot

function New-RerankerSecret {
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed or is not available in PATH. Install Docker Desktop first.'
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Engine is not running. Start Docker Desktop and run this script again.'
}

docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Compose v2 is required.'
}

$environmentPath = Join-Path $repositoryRoot '.env'
if (-not (Test-Path -LiteralPath $environmentPath)) {
    $environment = Get-Content -LiteralPath (Join-Path $repositoryRoot '.env.example') -Raw
    $environment = $environment.Replace('replace-with-external-secret', (New-RerankerSecret))
    $environment = $environment.Replace(
        'replace-with-different-external-secret',
        (New-RerankerSecret)
    )
    [System.IO.File]::WriteAllText($environmentPath, $environment)
    Write-Host 'Created .env with independent random API and admin secrets.'
}
else {
    Write-Host 'Keeping the existing .env file unchanged.'
}

$composeArguments = @('compose', 'up', '-d')
if (-not $SkipBuild) {
    $composeArguments += '--build'
}
& docker @composeArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Compose failed to start the service.'
}

$readyUri = 'http://localhost:8200/health/ready'
$deadline = (Get-Date).AddMinutes($ReadyTimeoutMinutes)
Write-Host 'Waiting for the pinned model to download, warm up, and become ready...'
do {
    try {
        $response = Invoke-RestMethod -Uri $readyUri -TimeoutSec 10
        if ($response.status -eq 'ready' -and $response.model_ready -eq $true) {
            Write-Host 'Reranker API is ready: http://localhost:8200'
            Write-Host 'Administration console: http://localhost:8400'
            exit 0
        }
    }
    catch {
        # A 503 response is expected while the pinned model downloads and warms up.
    }
    Start-Sleep -Seconds 5
} while ((Get-Date) -lt $deadline)

docker compose ps
docker compose logs --tail 50 reranker-api
throw "Readiness did not succeed within $ReadyTimeoutMinutes minutes."
