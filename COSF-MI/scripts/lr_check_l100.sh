#!/usr/bin/env bash
# l100 lr check for the new vanilla2 base (approved 2026-11-03; experiments/plans/vanilla2_base_10b.md): vanilla2 l100 at lr
# 6e-4 / 1e-3 / 1.5e-3 / 2e-3, 3,000 steps each (49M tokens), warmup 300, cosine to 0 at 3,000, seed 1, full v1 val at the end.
# Rule (fixed before running): the base uses the lr with the lowest val; ties within 0.005 go to the lower lr.
# STEPS=20 for a smoke test (writes to a scratch dir). Pause-aware, watchdog, FAILED marker via scripts/lib_train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
source scripts/lib_train.sh
STEPS=${STEPS:-3000}; WARM=$(( STEPS / 10 ))
ROOT=${OUT_ROOT:-checkpoints/lr_check_l100}
RES=${RES:-experiments/results/lr_check_l100}
mkdir -p "$ROOT" "$RES" run_control
echo "bash scripts/lr_check_l100.sh" > run_control/LAST_QUEUE
[ -f "$RES/results.tsv" ] || printf "lr\tval_loss\tmedian_tok_s\tpeak_gib\n" > "$RES/results.tsv"
for lr in 6e-4 1e-3 1.5e-3 2e-3; do
  out=$ROOT/lr_$lr
  if [ -f "$out/final_model.pt" ]; then echo "=== skip lr $lr (done)"; continue; fi
  mkdir -p "$out"
  echo "=== $(date -Iseconds) vanilla2 l100 lr $lr, $STEPS steps -> $out"
  run_train "$out/train.log" --model-size l100 --corpus data/corpus_v1/tok24k_v2 --seq-len 512 --batch-size 32 --lr "$lr" \
    --warmup "$WARM" --epochs 1 --max-steps "$STEPS" --decay-steps "$STEPS" --eval-batches 100000 --seed 1 --log-every 100 \
    --save-every 1000000 --save-final --resume auto --projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt \
    --checkpoint-dir "$out"
  val=$(grep -aoP 'val_loss=\K[0-9.]+' "$out/train.log" | tail -1 || true)
  tps=$(grep -aoP 'tok/s=\K[0-9]+' "$out/train.log" | tail -n +3 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}' || true)
  peak=$(grep -aoP 'peak VRAM \K[0-9.]+' "$out/train.log" | tail -1 || true)
  printf "%s\t%s\t%s\t%s\n" "$lr" "$val" "$tps" "$peak" >> "$RES/results.tsv"
  echo "    val $val, $tps tok/s"
  rm -f "$out/latest.pt" "$out/latest.prev.pt"   # only the final weights are needed
done
sort -t$'\t' -k2,2g "$RES/results.tsv" | head -3
echo "=== $(date -Iseconds) LR_CHECK_DONE"
