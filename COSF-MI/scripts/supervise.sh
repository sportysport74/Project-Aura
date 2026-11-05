#!/usr/bin/env bash
# Supervisor with limited auto-resume (user rule for the new base, 2026-11-04; same as 10B stage 2):
#   - only a watchdog stop ("no training progress for", scripts/lib_train.sh) is auto-resumed; a pause, a CUDA error, any other failure or a
#     PC-crash marker ends the supervisor, which waits for the user (run_control/FAILED is written).
#   - before each auto-resume: wait SUP_WAIT s (300), check nvidia-smi responds and the power limit reads 450 W.
#   - at most 2 auto-resumes; 2 faults within 3 h of each other → stop for good.
#   - each fault: append the Windows nvlddmkm/WHEA events, the trainer's GPU telemetry (<log>.gpu.csv) and the HWiNFO 10-min window
#     (experiments/hwinfo_fault_window.py on D:\fatba\Documents\train_hwinfo*) to $WATCH/CRASHES.md.
# Env: QUEUE (command, default "bash scripts/v2base_queue.sh"), WATCH (dir with the run's train logs, default checkpoints/v2base),
# DONE_MARK (default V2BASE_DONE), SUP_NAME (state/log name, default v2base; not NAME, which WSL sets to the host name), SUP_WAIT. Counters persist in run_control/<NAME>_supervisor.state.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
QUEUE=${QUEUE:-bash scripts/v2base_queue.sh}
WATCH=${WATCH:-checkpoints/v2base}
DONE_MARK=${DONE_MARK:-V2BASE_DONE}
NAME=${SUP_NAME:-v2base}
SUP_WAIT=${SUP_WAIT:-300}
STATE=run_control/${NAME}_supervisor.state
LOG=run_control/${NAME}_supervisor.log
mkdir -p run_control "$WATCH"
[ -f "$STATE" ] || echo "resumes=0 last_fault=0" > "$STATE"
# shellcheck disable=SC1090
source <(sed 's/ /\n/g' "$STATE")
say() { echo "=== $(date -Iseconds) [supervise:$NAME] $*" | tee -a "$LOG"; }
save_state() { echo "resumes=$resumes last_fault=$last_fault" > "$STATE"; }
stop() { say "STOP: $*. Waiting for the user."; mark_failed "supervise $NAME: $*"; exit 1; }

record_fault() {  # $1 = fault time (epoch), $2 = the stalled train.log
  local ft; ft=$(date -d "@$1" "+%Y-%m-%d %H:%M:%S")
  local hw; hw=$(ls -t /mnt/d/fatba/Documents/train_hwinfo* 2>/dev/null | head -1)
  {
    echo "## $(date -Iseconds): watchdog stop ($NAME), last progress $ft in $2"
    echo "- Windows nvlddmkm/WHEA/Display events in the 30 min before:"
    powershell.exe -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=([datetime]'$ft').AddMinutes(-30)} -ErrorAction SilentlyContinue | Where-Object { \$_.ProviderName -match 'nvlddmkm|WHEA|Display' } | ForEach-Object { '  - {0:yyyy-MM-dd HH:mm:ss} {1} {2} {3}' -f \$_.TimeCreated, \$_.ProviderName, \$_.Id, (((\$_.Properties | ForEach-Object { \$_.Value }) -join ' | ') -replace '^(.{0,120}).*','\$1') }" 2>/dev/null | tr -d '\r'
    echo "- GPU telemetry before the stall (last 12 samples of $2.gpu.csv):"
    head -1 "$2.gpu.csv" 2>/dev/null | sed 's/^/    /'; tail -n 12 "$2.gpu.csv" 2>/dev/null | sed 's/^/    /'
    if [ -n "$hw" ]; then $PY experiments/hwinfo_fault_window.py "$hw" --fault-time "$ft" --minutes 10 2>&1
    else echo "- No HWiNFO log found (D:\\fatba\\Documents\\train_hwinfo*)."; fi
    echo
  } >> "$WATCH/CRASHES.md"
}

while true; do
  qlog=run_control/${NAME}_attempt_$(date +%Y%m%d-%H%M%S).log
  say "launching (auto-resumes used: $resumes/2): $QUEUE -> $qlog"
  $QUEUE > "$qlog" 2>&1
  rc=$?
  echo "bash scripts/supervise.sh" > run_control/LAST_QUEUE
  cat "$qlog" >> "$LOG"
  if grep -q "$DONE_MARK" "$qlog"; then say "queue finished ($DONE_MARK)"; break
  elif grep -q "PAUSED on request" "$qlog"; then say "paused on request: supervisor stops (resume: scripts/resume_train.sh)"; exit 0
  elif grep -q "no training progress for" "$qlog"; then
    tl=$(find "$WATCH" -name train.log -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    fault=$(stat -c %Y "$tl")
    say "watchdog stop (last progress $(date -d "@$fault" +%H:%M:%S) in $tl): recording to $WATCH/CRASHES.md"
    record_fault "$fault" "$tl"
    if [ "$last_fault" -gt 0 ] && [ $(( fault - last_fault )) -lt $(( 3 * 3600 )) ]; then
      last_fault=$fault; save_state; stop "2 faults within 3 h, stopping for good"
    fi
    [ "$resumes" -ge 2 ] && { last_fault=$fault; save_state; stop "the 2 auto-resumes are used up"; }
    last_fault=$fault; save_state
    say "waiting ${SUP_WAIT} s before checking the GPU"; sleep "$SUP_WAIT"
    timeout 60 nvidia-smi > /dev/null 2>&1 || stop "nvidia-smi doesn't respond"
    cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
    [ "${cap%.*}" = 450 ] || stop "power limit ${cap} W, not 450 W"
    rm -f run_control/FAILED   # the watchdog's own FAILED note is superseded by this resume (kept in CRASHES.md and the log)
    resumes=$((resumes + 1)); save_state
    say "GPU responds, cap ${cap} W: auto-resume $resumes/2"
  else
    tail -n 5 "$qlog" | tee -a "$LOG"
    [ -f run_control/FAILED ] || mark_failed "supervise $NAME: queue exit $rc without a watchdog stop (see $qlog)"
    say "STOP: queue exit $rc and no watchdog stop: not auto-resumed. Waiting for the user."; exit 1
  fi
done
say "SUPERVISOR_DONE"
