#!/usr/bin/env bash
# After 10B stage 2 (called by scripts/stage2_supervisor.sh): the approved vanilla/embfix sweep (2026-11-01, user).
#   1. Diagnostic gate: s25 vanilla2 for 2,000 steps (same schedule as the sweep). Its loss at step 2000 must be <= 6.0 (the heartbeat-on
#      s25 runs are at 5.45 there; the broken vanilla was at 7.44). If it fails, the vanilla2 arms are skipped and embfix still runs.
#   2. Sweep: ARMS="vanilla2 embfix" (or "embfix") × s25/m50/l100 × seeds 1-3, through scripts/sweep_queue.sh (pause-aware, watchdog).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=checkpoints/sweep_diag/s25_vanilla2_s1
rm -rf "$D"; mkdir -p "$D"
echo "=== $(date -Iseconds) diagnostic: s25 vanilla2, 2000 steps -> $D"
$PY -u train_overnight.py --model-size s25 --corpus data/corpus_v1/tok24k_v2 --seq-len 512 --batch-size 32 --lr 6e-4 --warmup 500 \
  --epochs 1 --max-steps 2000 --decay-steps 12207 --seed 1 --log-every 200 --save-every 1000000 \
  --projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt --checkpoint-dir "$D" > "$D/train.log" 2>&1
loss=$(grep -aoP 'step=2000 loss=\K[0-9.]+' "$D/train.log" | tail -1)
echo "    vanilla2 loss at step 2000: ${loss:-none} (gate <= 6.0; heartbeat-on s25: 5.45; broken vanilla: 7.44)"
arms="embfix"
if [ -n "$loss" ] && awk -v l="$loss" 'BEGIN{exit !(l <= 6.0)}'; then arms="vanilla2 embfix"; else echo "    GATE FAILED: vanilla2 arms skipped"; fi
echo "=== $(date -Iseconds) sweep, arms: $arms"
ARMS="$arms" bash scripts/sweep_queue.sh
