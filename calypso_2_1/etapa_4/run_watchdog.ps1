# Supervisor anti-caigudes per a Calypso 2.1 (Etapa 4 encapsulada)
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

Write-Host "[WATCHDOG] Iniciant supervisor per calypso_2_1/serve.py des de $RepoRoot" -ForegroundColor Cyan

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Arrencant worker calypso-2.1 (calypso_2_1)..." -ForegroundColor Green
    python "calypso_2_1/serve.py" --poll-interval 0.12 --stock-interval 1.5 --live-interval 15.0
    $code = $LASTEXITCODE
    $tsEnd = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$tsEnd] Worker aturat (exit=$code). Reiniciant en 1s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}
