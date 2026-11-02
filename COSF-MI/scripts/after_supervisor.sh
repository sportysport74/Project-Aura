#!/usr/bin/env bash
# One-shot (2026-11-02): when scripts/stage2_supervisor.sh (pid $1) has exited with SUPERVISOR_DONE, check the 450 W cap and that no
# trainer runs, then launch scripts/next_queue.sh (aura_std sweep → ft_v2 curve) through launch_train.sh. Otherwise do nothing.
cd "$(dirname "$0")/.." || exit 1
while kill -0 "$1" 2>/dev/null; do sleep 60; done
grep -q "SUPERVISOR_DONE" run_control/stage2_supervisor.log || { echo "$(date -Iseconds) supervisor didn't finish cleanly: not launching"; exit 1; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || { echo "$(date -Iseconds) power limit ${cap} W, not 450: not launching"; exit 3; }
pgrep -f "^[^ ]*python[0-9.]* -u " > /dev/null && { echo "$(date -Iseconds) a python job is still running: not launching"; exit 1; }
echo "$(date -Iseconds) launching next_queue"
TRAIN_CMD="bash scripts/next_queue.sh" ~/.claude/skills/ml-training/scripts/launch_train.sh configs/smoke.yaml
