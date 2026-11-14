#!/usr/bin/env bash
# F6 · Supervisor: reinicia el servicio si muere. Ejecutar dentro de tmux/screen:
#   tmux new -s recsys './etapa_1_version_especial/run_forever.sh'
# Argumentos extra se pasan a serve.py (p. ej. --standby para la instancia de reserva).
cd "$(dirname "$0")"
mkdir -p logs
while true; do
  echo "$(date -u +%FT%TZ) arrancando serve.py $*" | tee -a logs/supervisor.log
  python -m src.serve "$@" 2>&1 | tee -a "logs/serve_$(date -u +%Y%m%d).log"
  echo "$(date -u +%FT%TZ) serve.py terminó con código ${PIPESTATUS[0]}; reinicio en 1 s" | tee -a logs/supervisor.log
  sleep 1
done
