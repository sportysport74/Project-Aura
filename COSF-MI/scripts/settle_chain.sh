#!/usr/bin/env bash
# Settling experiment chain (experiments/plans/settling_vanilla2.md): stage 1 = seed 1 of arms b-f; stage-2 selection
# (experiments/settle_decide.py, the pre-registered rule); stage 2 = seeds 2-3 for the selected arms; report. Pause-aware and watchdogged
# (sweep_queue.sh → run_train); any failure writes run_control/FAILED and stops. Relaunching continues (finished runs are skipped).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
export SETTLE_REPEATS=9   # compute-matched R4 (addendum 1)
me="bash scripts/settle_chain.sh"
phase() {  # $1 name, rest = command
  local name=$1; shift
  echo "=== $(date -Iseconds) [settle_chain] $name"
  "$@"; local rc=$?
  echo "$me" > run_control/LAST_QUEUE
  [ -f run_control/PAUSE ] && { echo "=== [settle_chain] paused during $name"; exit 0; }
  [ $rc -eq 0 ] || { [ -f run_control/FAILED ] || mark_failed "settle_chain phase '$name' failed (exit $rc)"; exit $rc; }
}
phase "stage 1 (seed 1)" env SIZES=s25 SEEDS=1 ARMS="settle_r4 settle_c settle_d settle_e settle_f" bash scripts/sweep_queue.sh
arms=$($PY experiments/settle_decide.py) || { mark_failed "settle_decide.py failed"; exit 1; }
cat experiments/results/settling/stage2_selection.md
if [ -n "$arms" ]; then
  phase "stage 2 (seeds 2-3: $arms)" env SIZES=s25 SEEDS="2 3" ARMS="$arms" bash scripts/sweep_queue.sh
fi
$PY experiments/sweep_report.py > /dev/null
echo "=== $(date -Iseconds) [settle_chain] SETTLE_CHAIN_DONE"
