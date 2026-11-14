# F6 · Supervisor para Windows. Dos procesos (4.5), cada uno en su terminal:
#   powershell -ExecutionPolicy Bypass -File etapa_2_version_especial\run_forever.ps1 serve
#   powershell -ExecutionPolicy Bypass -File etapa_2_version_especial\run_forever.ps1 builder
#   (domingo)  ...\run_forever.ps1 serve --freeze
# Antes: desactivar la suspensión con el portátil enchufado
#   powercfg /change standby-timeout-ac 0 ; powercfg /change hibernate-timeout-ac 0
param([string]$proc = "serve")
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs | Out-Null
while ($true) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "$stamp arrancando $proc $args" | Tee-Object -FilePath logs\supervisor.log -Append
    $log = "logs\$proc" + "_" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd") + ".log"
    python -m "src.$proc" @args 2>&1 | Tee-Object -FilePath $log -Append
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "$stamp $proc terminó (código $LASTEXITCODE); reinicio en 1 s" | Tee-Object -FilePath logs\supervisor.log -Append
    Start-Sleep -Seconds 1
}
