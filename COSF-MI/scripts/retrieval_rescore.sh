#!/usr/bin/env bash
# Retrieval re-score (pre-registered, experiments/plans/retrieval_rescore.md): the same arms, retrieval and seeds as the first eval, with the
# repetition penalty excluding source-only tokens (--rep-scope no_sources). One grading pass, then the report with the same rules.
# SMOKE=1: 2 questions, 1 sampled seed, into a scratch dir. Any stop → run_control/FAILED + ntfy alert; the end → alert.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
VPY=~/venvs/vllm/bin/python
F=${RES:-experiments/results/retrieval/factuality_rescore}
EVLIM=(); GRLIM=()
[ "${SMOKE:-0}" = 1 ] && { EVLIM=(--limit 2 --samples 1); GRLIM=(--limit 2); }
fail() { mark_failed "retrieval_rescore: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || fail "power limit ${cap} W, not 450 W"
[ "${SMOKE:-0}" = 1 ] || echo "bash scripts/retrieval_rescore.sh" > run_control/LAST_QUEUE
D=data/retrieval; CK=checkpoints/rag
B10S1=checkpoints/v2base/ft/b10_s1/best_model.pt; B10S0=checkpoints/v2base/ft/b10_s0/best_model.pt
mkdir -p "$F"
gen() { PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u experiments/factuality_gen.py --rep-scope no_sources "$@" --out "$F" \
          "${EVLIM[@]}" >> "$F/gen.log" 2>&1; }
echo "=== $(date -Iseconds) generation (rep-scope no_sources)"
: > "$F/gen.log"
gen R0="$B10S1" || fail "gen R0"
gen --append R0b="$B10S0" || fail "gen R0b"
gen --append --store "$D/corpus" R1c="$B10S1" || fail "gen R1c"
gen --append --store "$D/corpus" R2c_s0="$CK/s0/best_model.pt" R2c_s1="$CK/s1/best_model.pt" || fail "gen R2c"
gen --append --store "$D/wiki" R2w_s0="$CK/s0/best_model.pt" R2w_s1="$CK/s1/best_model.pt" || fail "gen R2w"
gen --append R2closed_s0="$CK/s0/best_model.pt" R2closed_s1="$CK/s1/best_model.pt" || fail "gen R2-closed"
gen --append --oracle R2oracle_s0="$CK/s0/best_model.pt" R2oracle_s1="$CK/s1/best_model.pt" || fail "gen R2-oracle"
echo "=== $(date -Iseconds) grading"
VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/factuality_grade.py "$F" "${GRLIM[@]}" > "$F/grade.log" 2>&1 || fail "grading failed"
$PY experiments/retrieval_report.py "$F" --recall-dir experiments/results/retrieval/recall || fail "report failed"
[ "${SMOKE:-0}" = 1 ] || bash scripts/notify.sh "Retrieval re-score done (sources excluded from the penalty): see factuality_rescore/report.md" 3
echo "=== $(date -Iseconds) RESCORE_DONE"
