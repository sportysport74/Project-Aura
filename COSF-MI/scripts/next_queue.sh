#!/usr/bin/env bash
# After the vanilla2/embfix sweep (2026-11-02, user): (1) the aura_std arm (s25/m50/l100 × 3 seeds), (2) scripts/ft_v2_queue.sh.
# Pause-aware: if a sub-queue was paused (run_control/PAUSE present), stop here, so a pause never skips ahead to the next phase.
# Relaunching this script (scripts/resume_train.sh does it via run_control/LAST_QUEUE) continues; finished runs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
me="bash scripts/next_queue.sh"
step() {  # $1 = name, rest = command
  local name=$1; shift
  echo "=== $(date -Iseconds) [next_queue] $name"
  "$@"; local rc=$?
  echo "$me" > run_control/LAST_QUEUE
  if [ -f run_control/PAUSE ]; then echo "=== [next_queue] paused during $name: stopping (resume: bash scripts/resume_train.sh)"; exit 0; fi
  if [ $rc -ne 0 ]; then echo "=== [next_queue] $name failed (exit $rc): stopping, not retrying"; exit $rc; fi
}
step "aura_std sweep" env ARMS=aura_std bash scripts/sweep_queue.sh
step "ft_v2 curve" bash scripts/ft_v2_queue.sh
echo "=== $(date -Iseconds) [next_queue] NEXT_QUEUE_DONE"
