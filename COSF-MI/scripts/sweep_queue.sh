#!/usr/bin/env bash
# Scaling sweep (scaling plan step 2, approved 2026-10-28): sizes s25 / m50 / l100 × heartbeat on / off / control × seeds 1-3,
# 200M tokens each on the clean corpus (data/corpus_v1/tok24k_v2), seq 512, batch 32 (16,384 tokens/step = 12,207 steps),
# lr 6e-4, warmup 500, cosine decayed to 0 at the step budget. Val loss on the whole corpus val split (645 batches).
# Data order is the same in every run (seeded per epoch), so seeds vary the init only and arms are paired by seed.
# Order: small to large; within a size, seed-major with the three arms interleaved.
# Stops on the first failure (set -e) and never retries. A CUDA error is reported as such.
# A run whose final_model.pt exists is skipped, so relaunching by hand continues where it stopped.
# Run via launch_train.sh with TRAIN_CMD="bash scripts/sweep_queue.sh".
# SIZES / SEEDS / ARMS override the grid, e.g. SIZES=l100 SEEDS="4 5 6" ARMS="on control" (extra seeds, same results.tsv).
# Arm "vanilla" (2026-10-31): --projection linear, the same model with every oscillator projection replaced by nn.Linear.
# Arms "vanilla2" (linear + embedding scale + GELU + GPT init: the fixed baseline) and "embfix" (heartbeat on + embedding scale),
# 2026-11-01, see experiments/plans/vanilla_embfix.md. Training goes through scripts/lib_train.sh: pause-aware (scripts/pause_train.sh / resume_train.sh) with a 15-min stall watchdog.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
source scripts/lib_train.sh
mkdir -p run_control
echo "env SIZES='${SIZES:-s25 m50 l100}' SEEDS='${SEEDS:-1 2 3}' ARMS='${ARMS:-on off control}' bash scripts/sweep_queue.sh" > run_control/LAST_QUEUE
RES=experiments/results/sweep/results.tsv
STEPS=12207
[ -f "$RES" ] || printf "size\tarm\tseed\tval_loss\tmedian_tok_s\tpeak_gib\tminutes\n" > "$RES"
for size in ${SIZES:-s25 m50 l100}; do
  for seed in ${SEEDS:-1 2 3}; do
    for arm in ${ARMS:-on off control}; do
      out=checkpoints/sweep/${size}_${arm}_s${seed}
      if [ -f "$out/final_model.pt" ]; then echo "=== skip ${size} ${arm} s${seed} (done)"; continue; fi
      mkdir -p "$out"
      case $arm in
        on) hb=(--heartbeat live) ;;
        off) hb=(--heartbeat live --heartbeat-strength 0) ;;
        control) hb=(--heartbeat live_control) ;;
        vanilla) hb=(--projection linear) ;;   # broken recipe (2026-10-31): kept for the record, don't rerun
        vanilla2) hb=(--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt) ;;
        embfix) hb=(--heartbeat live --emb-scale sqrt) ;;
        aura_std) hb=(--heartbeat live --emb-scale sqrt --ffn-activation gelu --linear-init gpt) ;;   # vanilla2's recipe, oscillator projections
        aura_std_off) hb=(--heartbeat live --heartbeat-strength 0 --emb-scale sqrt --ffn-activation gelu --linear-init gpt) ;;
      esac
      echo "=== $(date -Iseconds) ${size} ${arm} seed ${seed} -> ${out}"
      t0=$(date +%s)
      run_train "$out/train.log" --model-size "$size" --corpus data/corpus_v1/tok24k_v2 --seq-len 512 --batch-size 32 \
        --lr 6e-4 --warmup 500 --epochs 1 --max-steps $STEPS --decay-steps $STEPS --eval-batches 100000 --seed "$seed" \
        --log-every 200 --save-every 1000000 --save-final "${hb[@]}" --checkpoint-dir "$out"
      val=$(grep -oP 'val_loss=\K[0-9.]+' "$out/train.log" | tail -1 || true)
      tps=$(grep -oP 'tok/s=\K[0-9]+' "$out/train.log" | tail -n +5 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}' || true)
      peak=$(grep -oP 'peak VRAM \K[0-9.]+' "$out/train.log" | tail -1 || true)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$size" "$arm" "$seed" "$val" "$tps" "$peak" "$(( ($(date +%s) - t0) / 60 ))" >> "$RES"
      echo "    val ${val}, median ${tps} tok/s, peak ${peak} GiB"
    done
  done
  $PY experiments/sweep_report.py || true
done
echo "=== $(date -Iseconds) SWEEP_DONE"
