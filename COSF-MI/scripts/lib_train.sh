# shellcheck shell=bash
# Shared by the queue scripts (source it). run_train LOG ARGS...: runs train_overnight.py, appending to LOG.
#   exit 0  -> returns
#   exit 75 -> paused on request (scripts/pause_train.sh): the whole queue exits 0 here, so it doesn't move on to the next stage
#   other   -> STOPPED (a CUDA error is labelled), queue exits 1. Never retries.
# The queue must set PY and write its own relaunch command to run_control/LAST_QUEUE (scripts/resume_train.sh reads it).
# Stall watchdog: if the log hasn't changed for STALL_SEC (default 900 s) while the trainer is alive, it's hung (2026-10-31: after a
# GPU driver error, nvlddmkm Event 153, the CUDA context died and the trainer spun for 1h40 with no progress). The trainer is then
# killed and the queue STOPs (exit 1), never retried. tqdm writes to the log every step, so a live run touches it constantly.
run_train() {
  local log=$1; shift
  local rc=0
  $PY -u "${TRAIN_SCRIPT:-train_overnight.py}" "$@" >> "$log" 2>&1 &   # TRAIN_SCRIPT=finetune_chat.py for fine-tunes
  local pid=$!
  # GPU telemetry next to the log, every 30 s while the trainer runs (2026-11-01: stage 2 had none when it faulted)
  ( echo "time,mib,util,temp,watts,sm_mhz,mem_mhz,pcie_gen,pcie_width,event_reasons" > "$log.gpu.csv"
    while kill -0 "$pid" 2>/dev/null; do
      nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$(date +%F_%T),/" >> "$log.gpu.csv"; sleep 30
    done ) &
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$pid" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt "${STALL_SEC:-900}" ]; then
      echo "=== $(date -Iseconds) STOPPED: no training progress for ${STALL_SEC:-900} s (hung; likely a GPU driver error, check the"
      echo "    Windows System log for nvlddmkm events). Trainer killed, not retrying. Resume after a look: bash scripts/resume_train.sh"
      kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
      exit 1
    fi
  done
  wait "$pid" || rc=$?
  if [ "$rc" = 75 ]; then
    echo "=== $(date -Iseconds) PAUSED on request: $(grep -o 'PAUSED at step [0-9]*' "$log" | tail -1). Resume: bash scripts/resume_train.sh"
    exit 0
  elif [ "$rc" != 0 ]; then
    if grep -qE "CUDA error|AcceleratorError|CUDA_ERROR" "$log"; then
      echo "=== $(date -Iseconds) STOPPED: CUDA error (not retrying)"
    else
      echo "=== $(date -Iseconds) STOPPED: training failed with exit code $rc (not retrying)"
    fi
    tail -n 20 "$log"
    exit 1
  fi
}
