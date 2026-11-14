#!/usr/bin/env bash
# F6 · Supervisor (tmux). Dos procesos (4.5):
#   tmux new -s serve   './etapa_2_version_especial/run_forever.sh serve'
#   tmux new -s builder './etapa_2_version_especial/run_forever.sh builder'
cd "$(dirname "$0")"
proc="${1:-serve}"; shift || true
mkdir -p logs
while true; do
  echo "$(date -u +%FT%TZ) arrancando $proc $*" | tee -a logs/supervisor.log
  python -m "src.$proc" "$@" 2>&1 | tee -a "logs/${proc}_$(date -u +%Y%m%d).log"
  echo "$(date -u +%FT%TZ) $proc terminó con código ${PIPESTATUS[0]}; reinicio en 1 s" | tee -a logs/supervisor.log
  sleep 1
done
