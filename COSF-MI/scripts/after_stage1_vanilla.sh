#!/usr/bin/env bash
# Starts the vanilla baseline (sweep arm "vanilla": s25 / m50 / l100 × seeds 1-3, 200M tokens each) once 10B-pretrain stage 1 has
# really finished: experiments/results/pretrain_10b/gate_5b.json exists and no pretrain_10b_queue / trainer is running.
# It never launches over a running job, after a stage-1 STOP (no gate file appears), or if the power cap isn't 450 W (then it
# stops and says so). Gives up after 48 h. Run once in the background:
#   nohup bash scripts/after_stage1_vanilla.sh > run_control/after_stage1.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
GATE=experiments/results/pretrain_10b/gate_5b.json
end=$(( $(date +%s) + 48 * 3600 ))
echo "$(date -Iseconds) waiting for stage 1 ($GATE)"
until [ -f "$GATE" ] && ! pgrep -f "^bash scripts/pretrain_10b_queue.sh" > /dev/null \
      && ! pgrep -f "^[^ ]*python[0-9.]* -u (train_overnight|experiments/)" > /dev/null; do
  [ "$(date +%s)" -lt "$end" ] || { echo "$(date -Iseconds) gave up after 48 h"; exit 1; }
  sleep 120
done
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
if [ "${cap%.*}" != 450 ]; then echo "$(date -Iseconds) NOT STARTED: power limit ${cap} W, not 450 W"; exit 3; fi
echo "$(date -Iseconds) stage 1 done, cap ${cap} W: launching the vanilla baseline"
TRAIN_CMD="env ARMS=vanilla bash scripts/sweep_queue.sh" ~/.claude/skills/ml-training/scripts/launch_train.sh configs/smoke.yaml
