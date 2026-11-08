#!/usr/bin/env bash
# Settling retest, step 1 (floor check). Plan: experiments/plans/settling_retest.md (approved by Jon and frozen 2026-11-08).
#   vanilla2 s25, arithmetic only (data/arith_settle/tok24k), 200M tokens, seeds 1 and 2 -> eval on the fixed held-out test set
#   -> floor.md (primary cells = n >= 50 and vanilla2 mean in [20%, 60%]). The queue ENDS after step 1: Jon reads Addendum 1 first.
# SMOKE=1: 20 steps, 5 test problems per cell, scratch dirs; also builds and runs 20 steps of arms B, C, D (zero-init settling,
#   zero-init repeat, deeper vanilla2) and checks that B and C start as exact no-ops. Any stop -> run_control/FAILED + ntfy.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
DATA=data/arith_settle/tok24k
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then CK=checkpoints/smoke/settle_retest; RES=checkpoints/smoke/settle_retest/results; STEPS=20; LIMIT=(--limit 5)
else CK=checkpoints/settle_retest; RES=experiments/results/settle_retest; STEPS=12207; LIMIT=(); fi   # 12,207 × 32 × 512 ≈ 200M tokens
fail() { [ "$SMOKE" = 1 ] && { echo "=== SMOKE FAILED: $*"; exit 1; }; mark_failed "settle_retest: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }
note() { [ "$SMOKE" = 1 ] || bash scripts/notify.sh "$1" 3; echo "=== $(date -Iseconds) $1"; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || fail "power limit ${cap} W, not 450 W"
[ -f "$DATA/meta.json" ] || fail "no $DATA (build: scripts/corpus/build_arith_stream.py)"
[ "$SMOKE" = 1 ] || echo "bash scripts/settle_retest_queue.sh" > run_control/LAST_QUEUE
mkdir -p "$CK" "$RES"
# epochs needed for STEPS at 32 × 512 tokens per step (the stream repeats as epochs if it's shorter than 200M tokens)
EPOCHS=$($PY -c "import json,math;m=json.load(open('$DATA/meta.json'));print(max(1,math.ceil($STEPS*32*512/m['train_tokens'])))")
V2=(--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt)
run() {  # name seed extra-args...
  local name=$1 seed=$2; shift 2
  local out=$CK/$name
  if [ -f "$out/final_model.pt" ]; then echo "=== skip $name (done)"; return 0; fi
  mkdir -p "$out"
  echo "=== $(date -Iseconds) $name: seed $seed $*"
  run_train "$out/train.log" --model-size s25 "${V2[@]}" "$@" --corpus "$DATA" --seq-len 512 --batch-size 32 --lr 6e-4 --warmup 500 \
    --epochs "$EPOCHS" --max-steps $STEPS --decay-steps $STEPS --eval-batches 100000 --seed "$seed" --no-sleep --log-every 200 \
    --save-every 100000000 --save-every-minutes 30 --save-final --resume auto --checkpoint-dir "$out"
  [ -f "$out/final_model.pt" ] || fail "$name produced no final_model.pt ($out/train.log)"
  rm -f "$out/latest.pt" "$out/latest.prev.pt"
}
[ -f "$RES/test_set.jsonl" ] || $PY experiments/settle_arith_eval.py testset --out "$RES" "${LIMIT[@]}" > "$RES/testset.log" 2>&1 \
  || fail "test set build failed ($RES/testset.log)"
if [ "$SMOKE" = 1 ]; then   # arms B, C, D: build, no-op check at init, 20 steps
  $PY -m pytest -q tests/test_settling.py 2>&1 | tail -2 | grep -q "passed" || fail "unit tests (tests/test_settling.py)"
  run B_s1 1 --settle settle --settle-T 8 --settle-K 1.0 --settle-zero-init
  run C_s1 1 --settle repeat --settle-repeats 3 --settle-zero-init
  run D_s1 1 --n-layers 12
fi
run A_s1 1
run A_s2 2
$PY experiments/settle_arith_eval.py eval --out "$RES" A_s1="$CK/A_s1/final_model.pt" A_s2="$CK/A_s2/final_model.pt" > "$RES/eval_step1.log" 2>&1 \
  || fail "eval failed ($RES/eval_step1.log)"
$PY experiments/settle_arith_eval.py floor --out "$RES" A_s1 A_s2 > "$RES/floor.log" 2>&1; rc=$?
[ "$rc" = 0 ] || [ "$rc" = 3 ] || fail "floor report failed ($RES/floor.log)"
if [ "$SMOKE" = 1 ]; then
  $PY experiments/settle_arith_eval.py eval --out "$RES" B_s1="$CK/B_s1/final_model.pt" C_s1="$CK/C_s1/final_model.pt" \
    D_s1="$CK/D_s1/final_model.pt" > "$RES/eval_BCD.log" 2>&1 || fail "eval of B/C/D failed ($RES/eval_BCD.log)"
  echo "=== SMOKE OK"; exit 0
fi
if [ "$rc" = 3 ]; then note "settling retest step 1: fewer than 3 cells in the 20-60% range. Stopped per the plan; see $RES/floor.md"
else note "settling retest step 1 DONE: floor check in $RES/floor.md. Step 2 waits for Jon (Addendum 1)."; fi
echo "=== $(date -Iseconds) SETTLE_STEP1_DONE"
