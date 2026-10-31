#!/usr/bin/env bash
# Resume after scripts/pause_train.sh (or after a crash the user has acknowledged): checks the GPU power cap (must read 450 W; it
# resets on reboot) and that the GPU is free, clears the pause flag, then relaunches the queue recorded in run_control/LAST_QUEUE
# through launch_train.sh. The trainer resumes from latest.pt (weights, optimizer, scheduler, data position, RNG).
# Usage (from COSF-MI/): bash scripts/resume_train.sh          (ACK_CRASH=1 bash scripts/resume_train.sh after a recorded crash)
set -euo pipefail
cd "$(dirname "$0")/.."
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
if [ "${cap%.*}" != 450 ]; then
  echo "STOP: GPU power limit is ${cap} W, not 450 W. Re-apply it in an admin PowerShell: nvidia-smi -pl 450 (resets on reboot)." >&2
  exit 3
fi
if pgrep -f "^[^ ]*python[0-9.]* -u train_overnight.py" > /dev/null; then echo "A training run is already running; not starting another." >&2; exit 1; fi
[ -f run_control/LAST_QUEUE ] || { echo "No run_control/LAST_QUEUE: nothing to resume." >&2; exit 1; }
cmd=$(cat run_control/LAST_QUEUE)
rm -f run_control/PAUSE
echo "Power cap ${cap} W OK. Relaunching: $cmd"
TRAIN_CMD="env ACK_CRASH=${ACK_CRASH:-0} $cmd" ~/.claude/skills/ml-training/scripts/launch_train.sh configs/smoke.yaml
