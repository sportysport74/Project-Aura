#!/usr/bin/env bash
# Aura v3 base, steps 1-2 (approved 2026-11-07; rules pre-registered in experiments/plans/aura_v3_base.md):
#   1. scaling check: l100 (lr 6e-4) vs x330 (lr 3e-4), 200M tokens (3,052 steps × 65,536 tokens), seeds 1 and 2
#      → continue only if x330's val loss is lower on BOTH seeds; otherwise stop (alert) and report
#   2. lr check at x330, seed 1: 2e-4, 4.5e-4, 6e-4 (3e-4 = step 1's seed 1); if the winner is at a grid edge, one more point beyond it
# All runs: vanilla2, seq 1024, batch 16 × grad-accum 4, corpora v1 + ext (10B; one seeded permutation), cosine to 0 at the budget,
# val on the v1 val split (whole split). Through run_train (watchdog, telemetry, pause flag); each run resumes with --resume auto.
# SMOKE=1: 20 steps, 20 val batches, into a scratch dir. Any stop → run_control/FAILED + ntfy; gate verdicts and the end → ntfy.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
CK=${CKROOT:-checkpoints/v3_checks}
RES=${RES:-experiments/results/v3_checks}
STEPS=3052; EVB=100000
[ "${SMOKE:-0}" = 1 ] && { STEPS=20; EVB=20; }
fail() { mark_failed "v3_checks: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }
note() { [ "${SMOKE:-0}" = 1 ] || bash scripts/notify.sh "$1" 3; echo "=== $(date -Iseconds) $1"; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || fail "power limit ${cap} W, not 450 W"
[ "${SMOKE:-0}" = 1 ] || echo "bash scripts/v3_checks_queue.sh" > run_control/LAST_QUEUE
mkdir -p "$CK" "$RES"
V2=(--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt)
run() {  # name size lr seed
  local out=$CK/$1
  if [ -f "$out/final_model.pt" ]; then echo "=== skip $1 (done)"; return 0; fi
  mkdir -p "$out"
  echo "=== $(date -Iseconds) $1: $2 lr $3 seed $4"
  run_train "$out/train.log" --model-size "$2" "${V2[@]}" --corpus data/corpus_v1/tok24k_v2,data/corpus_v2/tok24k_ext \
    --val-corpus data/corpus_v1/tok24k_v2 --seq-len 1024 --batch-size 16 --grad-accum 4 --lr "$3" --warmup 300 --epochs 1 \
    --max-steps $STEPS --decay-steps $STEPS --eval-batches $EVB --seed "$4" --no-sleep --log-every 100 --save-every 100000000 \
    --save-every-minutes 30 --save-final --resume auto --checkpoint-dir "$out"
  [ -f "$out/final_model.pt" ] || fail "$1 produced no final_model.pt ($out/train.log)"
  $PY experiments/v3_checks_report.py --ck "$CK" --res "$RES" > /dev/null || fail "report failed"
  rm -f "$out/latest.pt" "$out/latest.prev.pt"   # full states (~4 GB at x330) aren't needed once the run is final
}
run scale_l100_s1 l100 6e-4 1 && run scale_x330_s1 x330 3e-4 1 && run scale_l100_s2 l100 6e-4 2 && run scale_x330_s2 x330 3e-4 2
v=$($PY experiments/v3_checks_report.py --ck "$CK" --res "$RES" --verdict scaling) || fail "report failed"
if [ "$v" != "PASS" ] && [ "${SMOKE:-0}" != 1 ]; then   # SMOKE: 20-step losses are noise; go on to exercise the lr phase
  note "v3 scaling check: x330 does NOT beat l100 on both seeds, stopping per the rule (see experiments/results/v3_checks/report.md)"
  exit 0
fi
note "v3 scaling check: $v (rule: x330 < l100 on both seeds); starting the lr check"
for lr in 2e-4 4.5e-4 6e-4; do run "lr_x330_$lr" x330 "$lr" 1; done
edge=$($PY experiments/v3_checks_report.py --ck "$CK" --res "$RES" --verdict lr_edge) || fail "report failed"
case $edge in
  low) run lr_x330_1.3e-4 x330 1.3e-4 1 ;;
  high) run lr_x330_9e-4 x330 9e-4 1 ;;
esac
$PY experiments/v3_checks_report.py --ck "$CK" --res "$RES" || fail "report failed"
note "v3 checks DONE (scaling + lr): see experiments/results/v3_checks/report.md. Step 3 waits for Jon."
echo "=== $(date -Iseconds) V3_CHECKS_DONE"
