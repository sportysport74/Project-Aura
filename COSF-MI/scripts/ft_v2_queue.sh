#!/usr/bin/env bash
# ft_v2 on the 10B-pretrain points (approved 2026-11-01/02): the real knowledge curve in chat. The same fine-tune (fact-first ft_v2 set,
# seq 512, batch 32, lr 1e-4, 1 epoch, replay sleep; the ft_l100 recipe) on the 2.2B / 5B / 10B bases, seeds 0 and 1, then chat-format
# factuality (both encodings, greedy + 3 sampled seeds, Qwen2.5-7B grader) and paired comparisons pooled over the 2 seeds.
# Fine-tunes go through run_train (stall watchdog, GPU log). A finished fine-tune (best_model.pt + DONE) is skipped on relaunch.
# Note: finetune_chat.py has no pause flag: pause_train.sh only pauses train_overnight runs. Killing a fine-tune loses that run (~48 min).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VPY=~/venvs/vllm/bin/python
source scripts/lib_train.sh
declare -A BASE=([b2]=checkpoints/pretrain/l100_on_s1/best_model.pt [b5]=checkpoints/pretrain/l100_5b_branch/final_model.pt
                 [b10]=checkpoints/pretrain/l100_10b/best_model.pt)
CK=()
for b in b2 b5 b10; do
  for seed in 0 1; do
    out=checkpoints/ft2/${b}_s${seed}
    CK+=("ft2_${b}_s${seed}=${out}/best_model.pt")
    if [ -f "$out/DONE" ]; then echo "=== skip ${b} s${seed} (done)"; continue; fi
    mkdir -p "$out"
    echo "=== $(date -Iseconds) ft_v2 on ${b} seed ${seed} -> ${out}"
    TRAIN_SCRIPT=finetune_chat.py run_train "$out/train.log" --checkpoint "${BASE[$b]}" --split-seed 0 --log-every 50 \
      --conversations 200000 --pilot data/distill_pilot/ft_v2 --eval-every 500 --seed "$seed" --seq-len 512 --out "$out"
    grep -a "^Done" "$out/train.log" | tail -1
    touch "$out/DONE"
  done
done
R=experiments/results/ft_v2_curve
mkdir -p "$R"
echo "=== $(date -Iseconds) factuality: generate"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u experiments/factuality_gen.py "${CK[@]}" --out "$R" > "$R/gen.log" 2>&1
echo "=== $(date -Iseconds) factuality: grade"
VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/factuality_grade.py "$R" > "$R/grade.log" 2>&1
for enc in joined pieces; do
  for pair in "b2 b5" "b5 b10" "b2 b10"; do
    set -- $pair
    $PY experiments/factuality_compare.py "$R" "ft2_$1_s0,ft2_$1_s1" "ft2_$2_s0,ft2_$2_s1" --encoding "$enc" --min-gain 0
  done
done | tee "$R/compare.txt"
echo "=== $(date -Iseconds) FT_V2_DONE"
