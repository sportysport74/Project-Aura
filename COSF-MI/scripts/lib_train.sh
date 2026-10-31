# shellcheck shell=bash
# Shared by the queue scripts (source it). run_train LOG ARGS...: runs train_overnight.py, appending to LOG.
#   exit 0  -> returns
#   exit 75 -> paused on request (scripts/pause_train.sh): the whole queue exits 0 here, so it doesn't move on to the next stage
#   other   -> STOPPED (a CUDA error is labelled), queue exits 1. Never retries.
# The queue must set PY and write its own relaunch command to run_control/LAST_QUEUE (scripts/resume_train.sh reads it).
run_train() {
  local log=$1; shift
  local rc=0
  $PY -u train_overnight.py "$@" >> "$log" 2>&1 || rc=$?
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
