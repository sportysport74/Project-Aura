#!/usr/bin/env bash
# GPU session monitor (2026-11-01): logs the card from WSL while something runs on Windows (e.g. a game), so it can be compared
# with training telemetry. Writes to runs/monitor_<timestamp>/:
#   gpu.csv     every 2 s: clocks (sm, graphics, mem), power draw + limit, temp, fan, util, P-state, PCIe gen/width, clock event reasons
#   events.txt  every 30 s: new Windows System log events from nvlddmkm, Display, WHEA-Logger, Kernel-Power, WER (bugchecks)
#   procs.csv   every 30 s: whether the watched Windows process is running (default PioneerGame = Arc Raiders)
# Stops after DURATION minutes (default 120) or when killed. CPU-only apart from the nvidia-smi queries.
# Usage (from COSF-MI/): nohup bash scripts/monitor_gpu_session.sh [minutes] [process_name] > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
DUR=${1:-120}; PROC=${2:-PioneerGame}
OUT=runs/monitor_$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
START=$(date -Iseconds)
echo "start=$START duration_min=$DUR watch_process=$PROC pid=$$" > "$OUT/meta.txt"
timeout "${DUR}m" nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit,clocks.sm,clocks.gr,clocks.mem,pstate,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active \
  --format=csv,nounits -lms 2000 > "$OUT/gpu.csv" 2>&1 &
SMI=$!
echo "time,running" > "$OUT/procs.csv"
end=$(( $(date +%s) + DUR * 60 ))
while [ "$(date +%s)" -lt "$end" ] && kill -0 $SMI 2>/dev/null; do
  r=$(powershell.exe -NoProfile -Command "if (Get-Process -Name '$PROC' -ErrorAction SilentlyContinue) { 1 } else { 0 }" 2>/dev/null | tr -d '\r')
  echo "$(date +%F_%T),${r:-?}" >> "$OUT/procs.csv"
  powershell.exe -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'$START'} -ErrorAction SilentlyContinue | Where-Object { \$_.ProviderName -match 'nvlddmkm|^Display$|WHEA|Kernel-Power|WER' } | ForEach-Object { '{0:yyyy-MM-dd HH:mm:ss} {1} {2} {3}' -f \$_.TimeCreated, \$_.ProviderName, \$_.Id, (((\$_.Properties | ForEach-Object { \$_.Value }) -join ' | ') -replace '^(.{0,160}).*','\$1') }" 2>/dev/null | tr -d '\r' > "$OUT/events.txt"
  sleep 30
done
kill $SMI 2>/dev/null
echo "end=$(date -Iseconds)" >> "$OUT/meta.txt"
