#!/usr/bin/env bash
# New AURA base (approved 2026-11-04; experiments/plans/vanilla2_base_10b.md): vanilla2 l100 on the same 10B tokens in the same order as
# the old curve, then ft_v2 × 2 seeds on each point. Phases (each skipped once done; an interrupted one resumes from its full state):
#   A   base: data/corpus_v1/tok24k_v2, 1 epoch (2.2B), lr 6e-4, warmup 2000, cosine to 0            → 2.2B point
#   B1  data/corpus_v2/tok24k_ext from A (--init-from), WSD peak 3e-4, warmup 2000, to step 140,380   → stable state
#   B2  5B branch: resume the stable state, linear decay to 0 over 30,517 steps                        → 5.0B point
#   B3  main run resumed to the end of the epoch, decayed over the last 10%                            → 10.0B point
#   E   evals: knowledge probe (new 2.2B/5B/10B; per-point vs the old curve; 10B vs 5B), corpus-format battery
#   F   ft_v2 on each point × seeds 0, 1; chat factuality (both encodings, greedy + 3 sampled); pooled comparisons, new vs old ft_v2 curve
# SMOKE=1: every phase for ~20 steps (evals with --limit 2) into CKROOT / RES scratch dirs, through the same code.
# Safety: 450 W cap check; QUEUE_RUNNING marker (a PC crash leaves it; relaunch needs ACK_CRASH=1); training via run_train (watchdog,
# pause, GPU log, run_control/FAILED). Supervised by scripts/supervise.sh (limited auto-resume).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
VPY=~/venvs/vllm/bin/python
SMOKE=${SMOKE:-0}
ROOT=${CKROOT:-checkpoints/v2base}
RES=${RES:-experiments/results/v2base}
OLD_PMI=experiments/results/pretrain_10b/knowledge_pmi_10b.json
OLD_FT=experiments/results/ft_v2_curve
MODEL=(--model-size l100 --projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt --seq-len 512 --batch-size 32
       --seed 1 --no-sleep --log-every 200 --save-every 100000000 --save-every-minutes 60)
S5=140380; B5=30517
LIM_A=(); LIM_B3=(); LIM_FT=(); LIM_EV=(); EVB=100000
if [ "$SMOKE" = 1 ]; then
  S5=20; B5=20; LIM_A=(--max-steps 20 --save-final); LIM_B3=(--max-steps 60 --save-final --eval-batches 5); LIM_FT=(--max-steps 20); LIM_EV=(--limit 2 --samples 1); EVB=5
fi
mkdir -p "$ROOT"/{A,main,branch5b,ft} "$RES" run_control
fail() { mark_failed "v2base: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }

# --- pre-flight ---------------------------------------------------------------------------------------------------------------
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || { mark_failed "power limit ${cap} W, not 450 W"; echo "STOPPED: power limit ${cap} W, not 450 W"; exit 3; }
MARK=$ROOT/QUEUE_RUNNING
if [ -f "$MARK" ]; then
  { echo "## Unclean end detected $(date -Iseconds)"; echo "- marker: $(cat "$MARK")"; echo; } >> "$ROOT/CRASHES.md"
  if [ "${ACK_CRASH:-0}" != 1 ]; then
    mark_failed "previous v2base run ended without a clean exit (PC crash or kill); relaunch with ACK_CRASH=1 after a look"
    echo "STOPPED: previous run did not end cleanly; recorded in $ROOT/CRASHES.md"; exit 2
  fi
fi
echo "started=$(date -Iseconds) pid=$$ smoke=$SMOKE" > "$MARK"
trap 'rm -f "$MARK"' EXIT
[ "$SMOKE" = 1 ] || echo "bash scripts/supervise.sh" > run_control/LAST_QUEUE

ckpt_of() { if [ -f "$1/best_model.pt" ]; then echo "$1/best_model.pt"; else echo "$1/final_model.pt"; fi; }

# --- A: 2.2B base ---------------------------------------------------------------------------------------------------------------
if [ ! -f "$ROOT/A/DONE" ]; then
  echo "=== $(date -Iseconds) A: base on tok24k_v2 -> $ROOT/A"
  run_train "$ROOT/A/train.log" "${MODEL[@]}" --corpus data/corpus_v1/tok24k_v2 --lr 6e-4 --warmup 2000 --epochs 1 --resume auto \
    --eval-batches "$EVB" "${LIM_A[@]}" --checkpoint-dir "$ROOT/A"
  [ -f "$(ckpt_of "$ROOT/A")" ] || fail "phase A produced no checkpoint"
  touch "$ROOT/A/DONE"
fi
B2_2=$(ckpt_of "$ROOT/A")
EXT=(--corpus data/corpus_v2/tok24k_ext --val-corpus data/corpus_v1/tok24k_v2 --lr 3e-4 --warmup 2000 --schedule wsd --decay-frac 0.1 --epochs 1)

# --- B1: to the stable state at S5 ------------------------------------------------------------------------------------------------
if [ ! -f "$ROOT/main/stable_step$S5.pt" ]; then
  echo "=== $(date -Iseconds) B1: main run to step $S5 -> $ROOT/main"
  run_train "$ROOT/main/train.log" "${MODEL[@]}" "${EXT[@]}" --init-from "$B2_2" --resume auto --max-steps "$S5" --checkpoint-dir "$ROOT/main"
  cp "$ROOT/main/latest.pt" "$ROOT/main/stable_step$S5.pt" || fail "B1: no latest.pt to keep as the stable state"
fi

# --- B2: 5B branch ---------------------------------------------------------------------------------------------------------------
if [ ! -f "$ROOT/branch5b/final_model.pt" ]; then
  echo "=== $(date -Iseconds) B2: 5B branch, steps $S5 -> $((S5 + B5)) -> $ROOT/branch5b"
  rs=(--resume-state "$ROOT/main/stable_step$S5.pt"); [ -f "$ROOT/branch5b/latest.pt" ] && rs=(--resume auto)
  run_train "$ROOT/branch5b/train.log" "${MODEL[@]}" "${EXT[@]}" "${rs[@]}" --decay-start-step "$S5" --decay-steps $((S5 + B5)) \
    --max-steps $((S5 + B5)) --save-final --eval-batches "$EVB" --checkpoint-dir "$ROOT/branch5b"
fi

# --- B3: main to 10B --------------------------------------------------------------------------------------------------------------
if [ ! -f "$ROOT/main/DONE" ]; then
  echo "=== $(date -Iseconds) B3: main run to the end of the epoch -> $ROOT/main"
  run_train "$ROOT/main/train.log" "${MODEL[@]}" "${EXT[@]}" --init-from "$B2_2" --resume auto "${LIM_B3[@]}" --checkpoint-dir "$ROOT/main"
  [ -f "$(ckpt_of "$ROOT/main")" ] || fail "phase B3 produced no checkpoint"
  touch "$ROOT/main/DONE"
fi
P2=$B2_2; P5=$ROOT/branch5b/final_model.pt; P10=$(ckpt_of "$ROOT/main")

# --- E: evals ---------------------------------------------------------------------------------------------------------------------
if [ ! -f "$RES/E_DONE" ]; then
  echo "=== $(date -Iseconds) E: knowledge probe + battery"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/knowledge_pmi.py b5="$P5" b10="$P10" b2="$P2" \
    --out "$RES/knowledge_pmi.json" > "$RES/knowledge_pmi.log" 2>&1 || fail "knowledge probe failed ($RES/knowledge_pmi.log)"
  $PY experiments/knowledge_compare.py --json old="$OLD_PMI" --json new="$RES/knowledge_pmi.json" \
    --pair old:base,new:b2 --pair old:b5,new:b5 --pair old:b10,new:b10 --pair new:b5,new:b10 --pair new:b2,new:b10 \
    --out "$RES/knowledge_compare.md" || fail "knowledge_compare failed"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/base_arith_eval.py b2="$P2" b5="$P5" b10="$P10" --max-new 160 \
    --out "$RES/arith" > "$RES/arith.log" 2>&1 || fail "battery failed ($RES/arith.log)"
  touch "$RES/E_DONE"
fi

# --- F: ft_v2 × 2 seeds on each point, chat factuality, new vs old --------------------------------------------------------------
declare -A BASE=([b2]=$P2 [b5]=$P5 [b10]=$P10)
CK=()
for b in b2 b5 b10; do
  for seed in 0 1; do
    out=$ROOT/ft/${b}_s${seed}
    CK+=("v2b_${b}_s${seed}=${out}/best_model.pt")
    [ -f "$out/DONE" ] && continue
    mkdir -p "$out"
    echo "=== $(date -Iseconds) F: ft_v2 on new ${b}, seed ${seed} -> ${out}"
    TRAIN_SCRIPT=finetune_chat.py run_train "$out/train.log" --checkpoint "${BASE[$b]}" --split-seed 0 --log-every 50 \
      --conversations 200000 --pilot data/distill_pilot/ft_v2 --eval-every 500 --seed "$seed" --seq-len 512 "${LIM_FT[@]}" --out "$out"
    [ -f "$out/best_model.pt" ] || fail "ft_v2 ${b} s${seed} produced no best_model.pt"
    touch "$out/DONE"
  done
done
F=$RES/factuality
mkdir -p "$F"
if [ ! -f "$F/grades.jsonl" ]; then
  echo "=== $(date -Iseconds) F: factuality generate + grade"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u experiments/factuality_gen.py "${CK[@]}" --out "$F" "${LIM_EV[@]}" \
    > "$F/gen.log" 2>&1 || fail "factuality_gen failed ($F/gen.log)"
  lim=(); [ "$SMOKE" = 1 ] && lim=(--limit 2)
  VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/factuality_grade.py "$F" "${lim[@]}" > "$F/grade.log" 2>&1 \
    || fail "factuality_grade failed ($F/grade.log)"
fi
C=$RES/factuality_vs_old
mkdir -p "$C"
cat "$OLD_FT/grades.jsonl" "$F/grades.jsonl" > "$C/grades.jsonl"
{
  for enc in joined pieces; do
    for b in b2 b5 b10; do
      $PY experiments/factuality_compare.py "$C" "ft2_${b}_s0,ft2_${b}_s1" "v2b_${b}_s0,v2b_${b}_s1" --encoding "$enc" --min-gain 0
    done
    $PY experiments/factuality_compare.py "$C" "v2b_b2_s0,v2b_b2_s1" "v2b_b10_s0,v2b_b10_s1" --encoding "$enc" --min-gain 0
  done
} > "$C/compare.txt" 2>&1 || fail "factuality comparison failed ($C/compare.txt)"
cat "$C/compare.txt"
echo "=== $(date -Iseconds) V2BASE_DONE"
