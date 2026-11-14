# Supervisor anti-caigudes per a Etapa 4 (calypso-3.2)
$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "[WATCHDOG] Iniciant supervisor per etapa_4/src/serve.py des de $ProjectRoot" -ForegroundColor Cyan

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Arrencant worker calypso-3.2 (Etapa 4)..." -ForegroundColor Green
    python "etapa_4/src/serve.py" --poll-interval 0.12 --stock-interval 1.5 --live-interval 15.0
    $code = $LASTEXITCODE
    $tsEnd = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$tsEnd] Worker aturat (exit=$code). Reiniciant en 1s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}
