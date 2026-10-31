#!/usr/bin/env bash
# Short queue for testing scripts/pause_train.sh and scripts/resume_train.sh (2026-10-31): l100 compiled, 600 steps of the v1
# corpus from the 2.2B checkpoint, with a "next stage" after training that must only run once training has really finished.
# Run via launch_train.sh with TRAIN_CMD="bash scripts/test_pause_queue.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=checkpoints/pause_test
mkdir -p "$OUT" run_control
source scripts/lib_train.sh
echo "bash scripts/test_pause_queue.sh" > run_control/LAST_QUEUE
echo "=== $(date -Iseconds) training -> $OUT"
run_train "$OUT/train.log" --model-size l100 --corpus data/corpus_v1/tok24k_v2 --seq-len 512 --batch-size 32 --lr 3e-4 \
  --warmup 50 --schedule wsd --epochs 1 --decay-steps 600 --max-steps 600 --seed 1 --heartbeat live --no-sleep --log-every 50 \
  --save-every 100000000 --save-every-minutes 60 --init-from checkpoints/pretrain/l100_on_s1/best_model.pt --resume auto \
  --checkpoint-dir "$OUT"
echo "=== $(date -Iseconds) NEXT STAGE ran (training finished)"
