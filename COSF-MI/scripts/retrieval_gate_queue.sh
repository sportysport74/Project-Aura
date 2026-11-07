#!/usr/bin/env bash
# Retrieval gate eval (pre-registered, experiments/plans/retrieval_gate.md). Phases: signals → thresholds (tune split only, frozen before any
# test item is read) → off-topic generation → off-topic grading (+ hand-check file) → multi-turn step A (generation + factuality grading) →
# answer entropy. It stops there: the report runs only after the 30-grade hand check (scripts don't read arm results before that).
# SMOKE=1: a few messages per phase, into a scratch dir. Any stop → run_control/FAILED + ntfy alert; the end → alert.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
VPY=~/venvs/vllm/bin/python
R=${RES:-experiments/results/retrieval/gate}
G=${GATE:-data/retrieval/gate/model}
SM=(); [ "${SMOKE:-0}" = 1 ] && SM=(--smoke)
fail() { mark_failed "retrieval_gate: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || fail "power limit ${cap} W, not 450 W"
[ "${SMOKE:-0}" = 1 ] || echo "bash scripts/retrieval_gate_queue.sh" > run_control/LAST_QUEUE
mkdir -p "$R"
ev() { PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u experiments/gate_eval.py "$1" --res "$R" --gate "$G" "${SM[@]}" >> "$R/eval.log" 2>&1; }
: > "$R/eval.log"
for ph in signals thresholds gen; do
  echo "=== $(date -Iseconds) $ph"
  ev $ph || fail "gate_eval $ph"
done
echo "=== $(date -Iseconds) grading off-topic replies"
VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/gate_grade.py "$R" > "$R/grade.log" 2>&1 || fail "gate_grade"
echo "=== $(date -Iseconds) multiturn"
ev multiturn || fail "gate_eval multiturn"
VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/factuality_grade.py "$R/multiturn" --questions "$R/multiturn/questions.jsonl" \
  > "$R/multiturn/grade.log" 2>&1 || fail "multiturn grading"
echo "=== $(date -Iseconds) entropy"
ev entropy || fail "gate_eval entropy"
[ "${SMOKE:-0}" = 1 ] || bash scripts/notify.sh "Gate eval generated + graded; hand check next (handcheck.md), then the report" 3
echo "=== $(date -Iseconds) GATE_EVAL_DONE"
