[CmdletBinding()]
param(
    [int]$ReadyTimeoutMinutes = 30,
    [switch]$SkipBuild,
    [switch]$Cpu
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

$composeFiles = @()
if ($Cpu) {
    $composeFiles = @('-f', 'docker-compose.cpu.yml')
}

$composePrefix = @('compose') + $composeFiles
if (-not $SkipBuild) {
    $exporterBuildArguments = $composePrefix + @('--profile', 'exporter', 'build', 'reranker-exporter')
    & docker @exporterBuildArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Compose failed to build the model exporter.'
    }
}

Write-Host 'Preparing or validating the pinned local ONNX artifact...'
$exportArguments = $composePrefix + @(
    '--profile', 'exporter', 'run', '--rm', 'reranker-exporter',
    '--model-id', 'BAAI/bge-reranker-v2-m3',
    '--revision', '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
    '--precision', 'fp32',
    '--score-transform', 'sigmoid'
)
& docker @exportArguments
if ($LASTEXITCODE -ne 0) {
    throw 'The pinned ONNX artifact could not be prepared or validated.'
}

$composeArguments = $composePrefix + @('up', '-d')
if (-not $SkipBuild) {
    $composeArguments += '--build'
}
& docker @composeArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Compose failed to start the service.'
}

$readyUri = 'http://localhost:8200/health/ready'
$deadline = (Get-Date).AddMinutes($ReadyTimeoutMinutes)
Write-Host 'Waiting for the local ONNX artifact to load, warm up, and become ready...'
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

$statusArguments = $composePrefix + @('ps')
$logArguments = $composePrefix + @('logs', '--tail', '50', 'reranker-api')
& docker @statusArguments
& docker @logArguments
throw "Readiness did not succeed within $ReadyTimeoutMinutes minutes."
