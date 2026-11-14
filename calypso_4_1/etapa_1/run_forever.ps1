# F6 · Supervisor para Windows (PowerShell). Reinicia el servicio si muere.
#   powershell -ExecutionPolicy Bypass -File etapa_1_version_especial\run_forever.ps1
#   powershell -ExecutionPolicy Bypass -File etapa_1_version_especial\run_forever.ps1 --standby
# Antes: Configuración > Sistema > Energía > "Suspender: Nunca" con el portátil enchufado
# (o:  powercfg /change standby-timeout-ac 0 ; powercfg /change hibernate-timeout-ac 0)
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs | Out-Null
while ($true) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "$stamp arrancando serve.py $args" | Tee-Object -FilePath logs\supervisor.log -Append
    $log = "logs\serve_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd") + ".log"
    python -m src.serve @args 2>&1 | Tee-Object -FilePath $log -Append
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "$stamp serve.py terminó (código $LASTEXITCODE); reinicio en 1 s" | Tee-Object -FilePath logs\supervisor.log -Append
    Start-Sleep -Seconds 1
}
