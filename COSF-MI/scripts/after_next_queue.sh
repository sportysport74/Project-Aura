#!/usr/bin/env bash
# One-shot (2026-11-02): when scripts/next_queue.sh (launched as pid $1) has exited and its log says NEXT_QUEUE_DONE, check the 450 W cap
# and that no python job runs, then launch scripts/heartbeat_retest_queue.sh through launch_train.sh. Otherwise do nothing.
cd "$(dirname "$0")/.." || exit 1
LOG=$2
while kill -0 "$1" 2>/dev/null; do sleep 60; done
grep -q "NEXT_QUEUE_DONE" "$LOG" || { echo "$(date -Iseconds) next_queue didn't finish cleanly: not launching"; exit 1; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || { echo "$(date -Iseconds) power limit ${cap} W, not 450: not launching"; exit 3; }
pgrep -f "^[^ ]*python[0-9.]* -u " > /dev/null && { echo "$(date -Iseconds) a python job is still running: not launching"; exit 1; }
echo "$(date -Iseconds) launching the heartbeat retest"
TRAIN_CMD="bash scripts/heartbeat_retest_queue.sh" ~/.claude/skills/ml-training/scripts/launch_train.sh configs/smoke.yaml
