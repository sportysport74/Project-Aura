#!/usr/bin/env bash
# Scaling plan step 0 + the decoding check, back to back on the GPU (2026-10-28):
#   (order since the 2026-10-28 crash: arithmetic first, decoding check last with --bs 16)
#   1. experiments/decoding_check.py on the distill A/B checkpoints (top-k 40 / top-p 0.9 / top-p 0.9 + rep 1.15, 2 samples)
#   2. arithmetic-format test: fine-tune ab_new_s0 on data/arith_format/{plain,scratch}, seeds 0 and 1 (interleaved)
#   3. experiments/ab_eval.py on the four arithmetic checkpoints (battery, readable set, val sets, exam)
# Run via launch_train.sh with TRAIN_CMD="bash scripts/step0_queue.sh" (the config argument is just recorded).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CKPTS=()
for seed in 0 1; do
  for fmt in plain scratch; do
    out=checkpoints/arith_${fmt}_s${seed}
    mkdir -p "$out"
    echo "=== $(date -Iseconds) arith ${fmt} seed ${seed} -> ${out}"
    $PY -u finetune_chat.py --checkpoint checkpoints/ab_new_s0/best_model.pt --pilot "data/arith_format/${fmt}" \
      --split-seed 0 --seed "$seed" --eval-every 1000 --log-every 100 --out "$out" > "$out/train.log" 2>&1
    tail -n 2 "$out/train.log"
    CKPTS+=("arith_${fmt}_s${seed}=${out}/best_model.pt")
  done
done
echo "=== $(date -Iseconds) eval"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/ab_eval.py --out experiments/results/arith_format "${CKPTS[@]}"
echo "=== $(date -Iseconds) decoding check"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/decoding_check.py --bs 16 --out experiments/results/ab_distill/decoding \
  new_s0=checkpoints/ab_new_s0/best_model.pt new_s1=checkpoints/ab_new_s1/best_model.pt \
  old_s0=checkpoints/ab_old_s0/best_model.pt old_s1=checkpoints/ab_old_s1/best_model.pt \
  ref_aura_chat_distill=checkpoints/aura_chat_distill/best_model.pt
echo "=== $(date -Iseconds) STEP0_DONE"
