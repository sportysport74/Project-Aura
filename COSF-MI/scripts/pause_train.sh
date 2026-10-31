#!/usr/bin/env bash
# Pause the running training (e.g. to game): asks train_overnight.py to save its full state (latest.pt) and exit cleanly,
# then waits until it has. The queue it ran in stops without starting its next stage. Resume with scripts/resume_train.sh.
# If the queue is in an eval phase (no train_overnight.py running), nothing is paused; it says so and leaves the flag unset.
# Usage (from COSF-MI/): bash scripts/pause_train.sh [timeout_s=600]
set -euo pipefail
cd "$(dirname "$0")/.."
TIMEOUT=${1:-600}
pids=$(pgrep -f "^[^ ]*python[0-9.]* -u train_overnight.py" || true)
if [ -z "$pids" ]; then
  echo "No train_overnight.py is running, nothing to pause."
  ps -eo pid,etime,cmd | grep -E "factuality_|base_arith_eval|vllm" | grep -v grep || true
  echo "If an eval is running above, wait for it (or kill it; evals are cheap to rerun). The GPU is otherwise free."
  exit 0
fi
mkdir -p run_control
date -Iseconds > run_control/PAUSE
echo "Pause requested for pid(s) $pids; waiting for the checkpoint save (up to ${TIMEOUT}s)..."
t0=$(date +%s)
while pgrep -f "^[^ ]*python[0-9.]* -u train_overnight.py" > /dev/null; do
  if [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT" ]; then
    echo "Still running after ${TIMEOUT}s (stuck in a val pass or a long save?). Flag left in place; check the log." >&2
    exit 1
  fi
  sleep 2
done
log=$(ls -t checkpoints/*/*/train.log checkpoints/*/train.log 2>/dev/null | head -1)
grep -o "PAUSED at step [0-9]*: latest.pt saved in [^;]*" "$log" | tail -1 || echo "(no PAUSED line in $log: it may have ended for another reason)"
sleep 3
echo "GPU now: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader). Safe to game."
