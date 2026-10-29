#!/usr/bin/env bash
# × / ÷ data-vs-capacity test (2026-10-28): same steps, 25k unique problems × 4 epochs vs 100k × 1 epoch, seeds 0 and 1,
# from arith_plain_s0 (step 0's best). Then experiments/arith_md_eval.py (fit vs held-out per digit cell + battery).
# Run via launch_train.sh with TRAIN_CMD="bash scripts/arith_md_queue.sh". Stops on any error (set -e), no retry.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CKPTS=("base_arith_plain_s0=checkpoints/arith_plain_s0/best_model.pt")
for seed in 0 1; do
  for arm in n25k:4 n100k:1; do
    data=${arm%%:*}; ep=${arm##*:}; out=checkpoints/arith_md_${data}_s${seed}
    mkdir -p "$out"
    echo "=== $(date -Iseconds) ${data} x${ep} epochs seed ${seed} -> ${out}"
    $PY -u finetune_chat.py --checkpoint checkpoints/arith_plain_s0/best_model.pt --pilot "data/arith_md/${data}" \
      --epochs "$ep" --split-seed 0 --seed "$seed" --eval-every 1000 --log-every 200 --out "$out" > "$out/train.log" 2>&1
    tail -n 1 "$out/train.log"
    CKPTS+=("${data}_s${seed}=${out}/best_model.pt")
  done
done
echo "=== $(date -Iseconds) eval"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/arith_md_eval.py --out experiments/results/arith_md "${CKPTS[@]}"
echo "=== $(date -Iseconds) ARITH_MD_DONE"
