#!/usr/bin/env bash
# Night queue (planned 2026-10-29): (1) full pretrain of l100 with the heartbeat on, one epoch of the clean corpus
# (data/corpus_v1/tok24k_v2, 2.2B tokens = 134,266 steps of 32 x 512), the sweep's settings except warmup 2000, sleep off
# (the sweep never slept; see --sleep in train_overnight.py); (2) 3 more seeds (4-6) of l100 on vs control at 200M tokens,
# appended to the sweep's results.tsv so the report pairs 6 seeds.
# Stops on the first failure (set -e) and never retries. A CUDA error is reported as such. A Windows crash simply ends it.
# Relaunching by hand continues: the pretrain resumes from latest.pt (--resume auto), finished sweep runs are skipped.
# Run via launch_train.sh with TRAIN_CMD="bash scripts/pretrain_queue.sh".
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
out=checkpoints/pretrain/l100_on_s1
mkdir -p "$out"
if [ -f "$out/best_model.pt" ]; then
  echo "=== skip pretrain (done)"
else
  echo "=== $(date -Iseconds) pretrain l100 on seed 1 -> ${out}"
  if ! $PY -u train_overnight.py --model-size l100 --corpus data/corpus_v1/tok24k_v2 --seq-len 512 --batch-size 32 \
      --lr 6e-4 --warmup 2000 --epochs 1 --seed 1 --log-every 200 --save-every 3000 --no-sleep \
      --heartbeat live --resume auto --checkpoint-dir "$out" >> "$out/train.log" 2>&1; then
    if grep -qE "CUDA error|AcceleratorError|CUDA_ERROR" "$out/train.log"; then
      echo "=== $(date -Iseconds) STOPPED: CUDA error in pretrain (not retrying)"
    else
      echo "=== $(date -Iseconds) STOPPED: pretrain failed (not retrying)"
    fi
    tail -n 20 "$out/train.log"
    exit 1
  fi
  grep -E "Val loss|peak VRAM" "$out/train.log" | tail -2
fi
echo "=== $(date -Iseconds) extra seeds: l100 on/control seeds 4-6"
SIZES=l100 SEEDS="4 5 6" ARMS="on control" bash scripts/sweep_queue.sh
echo "=== $(date -Iseconds) NIGHT_DONE"
