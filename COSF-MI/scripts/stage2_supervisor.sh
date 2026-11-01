#!/usr/bin/env bash
# Stage-2 supervisor for the 10B pretrain (user decision 2026-11-01, THIS RUN ONLY): runs `STAGE=to10b pretrain_10b_queue.sh` with a
# limited auto-resume after GPU hangs.
#   - Only a watchdog stop ("no training progress", lib_train.sh) is auto-resumed. A pause, a CUDA error, any other failure, or a PC
#     crash (the queue's QUEUE_RUNNING marker, exit 2) ends the supervisor, which then waits for the user.
#   - Before each auto-resume: wait 5 min, check that nvidia-smi responds and the power limit reads 450 W.
#   - At most 2 auto-resumes. If 2 faults happen within 3 h of each other, stop for good.
#   - On each fault: append the Windows nvlddmkm events and the HWiNFO rows of the 10 min before it
#     (experiments/hwinfo_fault_window.py on D:\fatba\Documents\train_hwinfo*) to checkpoints/pretrain/l100_10b/CRASHES.md.
#   - After a clean finish: the 10B knowledge probe, then scripts/post_stage2.sh if it exists (the approved sweep).
# Counters live in run_control/stage2_supervisor.state, so a pause + scripts/resume_train.sh (which relaunches this supervisor via
# run_control/LAST_QUEUE) keeps them.
# Launch: TRAIN_CMD="bash scripts/stage2_supervisor.sh" ~/.claude/skills/ml-training/scripts/launch_train.sh configs/smoke.yaml
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
MAIN=checkpoints/pretrain/l100_10b
STATE=run_control/stage2_supervisor.state
SUP_LOG=run_control/stage2_supervisor.log
mkdir -p run_control
[ -f "$STATE" ] || echo "resumes=0 last_fault=0" > "$STATE"
# shellcheck disable=SC1090
source <(sed 's/ /\n/g' "$STATE")
say() { echo "=== $(date -Iseconds) [supervisor] $*" | tee -a "$SUP_LOG"; }
save_state() { echo "resumes=$resumes last_fault=$last_fault" > "$STATE"; }

record_fault() {  # $1 = fault time (epoch, last training progress)
  local ft; ft=$(date -d "@$1" "+%Y-%m-%d %H:%M:%S")
  local hw; hw=$(ls -t /mnt/d/fatba/Documents/train_hwinfo* 2>/dev/null | head -1)
  {
    echo "## $(date -Iseconds) — watchdog stop in stage 2 (supervisor), last progress $ft"
    echo "- Windows nvlddmkm/WHEA events in the 30 min before:"
    powershell.exe -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=([datetime]'$ft').AddMinutes(-30)} -ErrorAction SilentlyContinue | Where-Object { \$_.ProviderName -match 'nvlddmkm|WHEA|Display' } | ForEach-Object { '  - {0:yyyy-MM-dd HH:mm:ss} {1} {2} {3}' -f \$_.TimeCreated, \$_.ProviderName, \$_.Id, (((\$_.Properties | ForEach-Object { \$_.Value }) -join ' | ') -replace '^(.{0,120}).*','\$1') }" 2>/dev/null | tr -d '\r'
    echo "- nvidia-smi samples before the stall (run_train's .gpu.csv, last 10):"
    tail -n 10 "$MAIN/train.log.gpu.csv" 2>/dev/null | sed 's/^/    /'
    if [ -n "$hw" ]; then
      $PY experiments/hwinfo_fault_window.py "$hw" --fault-time "$ft" --minutes 10 2>&1
    else
      echo "- **No HWiNFO log found** (D:\\fatba\\Documents\\train_hwinfo*)."
    fi
    echo
  } >> "$MAIN/CRASHES.md"
}

while true; do
  echo "bash scripts/stage2_supervisor.sh" > run_control/LAST_QUEUE.supervisor
  log=run_control/stage2_attempt_$(date +%Y%m%d-%H%M%S).log
  say "launching stage 2 (auto-resumes used: $resumes/2) -> $log"
  env STAGE=to10b bash scripts/pretrain_10b_queue.sh > "$log" 2>&1
  rc=$?
  # the queue writes its own LAST_QUEUE; point resume_train.sh back at this supervisor
  cp run_control/LAST_QUEUE.supervisor run_control/LAST_QUEUE
  cat "$log" >> "$SUP_LOG"
  if grep -q "STAGE to10b DONE" "$log"; then
    say "stage 2 finished cleanly"
    break
  elif grep -q "PAUSED on request" "$log"; then
    say "paused on request: supervisor stops. Resume with scripts/resume_train.sh (relaunches the supervisor, counters kept)."
    exit 0
  elif grep -q "no training progress for" "$log"; then
    fault=$(stat -c %Y "$MAIN/train.log")
    say "watchdog stop (last progress $(date -d "@$fault" +%H:%M:%S)); recording to CRASHES.md"
    record_fault "$fault"
    if [ "$last_fault" -gt 0 ] && [ $(( fault - last_fault )) -lt $(( 3 * 3600 )) ]; then
      say "STOP FOR GOOD: 2 faults within 3 h ($(date -d "@$last_fault" +%H:%M) and $(date -d "@$fault" +%H:%M)). Waiting for the user."
      last_fault=$fault; save_state; exit 1
    fi
    if [ "$resumes" -ge 2 ]; then
      say "STOP FOR GOOD: the 2 auto-resumes are used up. Waiting for the user."
      last_fault=$fault; save_state; exit 1
    fi
    last_fault=$fault; save_state
    say "waiting 5 min before checking the GPU"
    sleep 300
    if ! timeout 60 nvidia-smi > /dev/null 2>&1; then say "STOP: nvidia-smi doesn't respond. Waiting for the user."; exit 1; fi
    cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
    if [ "${cap%.*}" != 450 ]; then say "STOP: power limit ${cap} W, not 450 W. Waiting for the user."; exit 3; fi
    resumes=$((resumes + 1)); save_state
    say "GPU responds, cap ${cap} W: auto-resume $resumes/2 from latest.pt"
  else
    say "STOP: the queue ended with exit $rc and no watchdog stop (CUDA error, failure, PC-crash marker or cap): not auto-resumed. Last lines:"
    tail -n 5 "$log" | tee -a "$SUP_LOG"
    exit 1
  fi
done

say "10B knowledge probe"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/knowledge_pmi.py b5=checkpoints/pretrain/l100_5b_branch/final_model.pt \
  b10=$MAIN/best_model.pt base=checkpoints/pretrain/l100_on_s1/best_model.pt \
  --out experiments/results/pretrain_10b/knowledge_pmi_10b.json 2>&1 | grep -v Warning | tee -a "$SUP_LOG"
if [ -f scripts/post_stage2.sh ]; then
  say "running scripts/post_stage2.sh"
  bash scripts/post_stage2.sh 2>&1 | tee -a "$SUP_LOG"
fi
say "SUPERVISOR_DONE"
