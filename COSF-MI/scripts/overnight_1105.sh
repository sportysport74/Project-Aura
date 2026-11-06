#!/usr/bin/env bash
# Overnight 2026-11-05 (experiments/plans/overnight_1105.md): (1) phase-A-style cosine anneal from the 4.5B stable state + knowledge probe
# vs 2.2B/5B/10B; (2) l100 lr check at 3e-4. SMOKE=1: 20-step anneal into scratch, the probe on it, a 20-step lr check.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
SMOKE=${SMOKE:-0}
ROOT=${CKROOT:-checkpoints/v2base}
RES=${RES:-experiments/results/v2base_anneal}
STABLE=checkpoints/v2base/main/stable_step140380.pt
S=140380; N=134266; W=2000; EVB=100000
[ "$SMOKE" = 1 ] && { N=20; W=5; EVB=5; }
fail() { mark_failed "overnight_1105: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || fail "power limit ${cap} W, not 450 W"
mkdir -p "$ROOT/anneal45_cos" "$RES" run_control
[ "$SMOKE" = 1 ] || echo "bash scripts/overnight_1105.sh" > run_control/LAST_QUEUE
A=$ROOT/anneal45_cos
if [ ! -f "$A/final_model.pt" ]; then
  echo "=== $(date -Iseconds) anneal: cosine from 6e-4 over $N steps from the 4.5B stable state -> $A"
  rs=(--resume-state "$STABLE"); [ -f "$A/latest.pt" ] && rs=(--resume auto)
  run_train "$A/train.log" --model-size l100 --projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt --seq-len 512 \
    --batch-size 32 --seed 1 --no-sleep --log-every 200 --save-every 100000000 --save-every-minutes 60 \
    --corpus data/corpus_v2/tok24k_ext --val-corpus data/corpus_v1/tok24k_v2 --epochs 1 "${rs[@]}" \
    --lr 6e-4 --warmup "$W" --branch-cosine-steps "$N" --branch-start-step "$S" --max-steps $((S + N)) --save-final \
    --eval-batches "$EVB" --checkpoint-dir "$A"
  [ -f "$A/final_model.pt" ] || fail "anneal produced no final_model.pt"
fi
grep -aoE "val_loss=[0-9.]+" "$A/train.log" | tail -1
echo "=== $(date -Iseconds) knowledge probe: anneal vs 2.2B / 5B / 10B"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/knowledge_pmi.py b10=checkpoints/v2base/main/best_model.pt \
  anneal="$A/final_model.pt" b2=checkpoints/v2base/A/best_model.pt b5=checkpoints/v2base/branch5b/final_model.pt \
  --out "$RES/knowledge_pmi.json" > "$RES/knowledge_pmi.log" 2>&1 || fail "probe failed ($RES/knowledge_pmi.log)"
$PY experiments/knowledge_compare.py --json new="$RES/knowledge_pmi.json" --pair new:b10,new:anneal --pair new:b2,new:anneal \
  --pair new:b5,new:anneal --pair new:b2,new:b10 --out "$RES/knowledge_compare.md" || fail "knowledge_compare failed"
echo "=== $(date -Iseconds) l100 lr check at 3e-4"
if [ "$SMOKE" = 1 ]; then
  STEPS=20 LRS=3e-4 OUT_ROOT="$ROOT/lrsmoke" RES="$RES/lrsmoke" bash scripts/lr_check_l100.sh || fail "lr check (smoke) failed"
else
  LRS=3e-4 bash scripts/lr_check_l100.sh || fail "lr check failed"
fi
echo "=== $(date -Iseconds) OVERNIGHT_1105_DONE"
