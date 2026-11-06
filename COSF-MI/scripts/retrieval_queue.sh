#!/usr/bin/env bash
# Retrieval experiment (approved 2026-11-06; pre-registered in experiments/plans/retrieval_prereg.md). Phases (each skipped once done):
#   1 build stores (CPU): corpus (Cosmopedia + FineWeb-Edu, cap 6M passages) and wiki (Simple English Wikipedia, inference only)
#   2 embed both stores (bge-small, GPU)
#   3 recall@3 on the 100 questions for both stores (Qwen grader); GATE: corpus recall@3 >= 0.6, else stop
#   4 RAG fine-tune data from the CORPUS store only (retrieve + Qwen teacher) → data/distill_pilot/ft_rag
#   5 RAG fine-tune from the 10B base, seeds 0 and 1
#   6 factuality generation for every arm (R0, R0b, R1c, R2c, R2w, R2-closed, R2-oracle), one grading run, report with the rules applied
# SMOKE=1: tiny stores, 5-question recall (no gate), ~200 teacher requests, 20-step fine-tunes, 2-question evals, all in scratch.
# Training through run_train (watchdog, GPU log, pause); any stop → run_control/FAILED + phone alert (ntfy); the end → alert. No auto-resume.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
PY=.venv/bin/python
VPY=~/venvs/vllm/bin/python
SMOKE=${SMOKE:-0}
D=${DROOT:-data/retrieval}
FT=${FTDATA:-data/distill_pilot/ft_rag}
CK=${CKROOT:-checkpoints/rag}
RES=${RES:-experiments/results/retrieval}
BASE10=checkpoints/v2base/main/best_model.pt
B10S1=checkpoints/v2base/ft/b10_s1/best_model.pt
B10S0=checkpoints/v2base/ft/b10_s0/best_model.pt
MAXDOCS=(); RLIM=(); GLIM=(); FTLIM=(); EVLIM=(); GRLIM=()
if [ "$SMOKE" = 1 ]; then
  MAXDOCS=(--max-docs 2000); RLIM=(--limit 300); GLIM=(--limit 200); FTLIM=(--max-steps 20); EVLIM=(--limit 2 --samples 1); GRLIM=(--limit 2)
fi
fail() { mark_failed "retrieval: $*"; echo "=== $(date -Iseconds) STOPPED: $*"; exit 1; }
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
[ "${cap%.*}" = 450 ] || fail "power limit ${cap} W, not 450 W"
mkdir -p "$D" "$CK" "$RES" run_control
[ "$SMOKE" = 1 ] || echo "bash scripts/retrieval_queue.sh" > run_control/LAST_QUEUE

for s in corpus wiki; do
  if [ ! -f "$D/$s/passages.jsonl" ]; then
    echo "=== $(date -Iseconds) 1: build the $s store"
    $PY scripts/retrieval/build_store.py "$s" --out "$D/$s" "${MAXDOCS[@]}" > "$RES/build_$s.log" 2>&1 || fail "build_store $s failed"
    tail -1 "$RES/build_$s.log" | cut -c1-200
  fi
  if [ ! -f "$D/$s/emb.npy" ]; then
    echo "=== $(date -Iseconds) 2: embed the $s store"
    $PY scripts/retrieval/embed_store.py "$D/$s" > "$RES/embed_$s.log" 2>&1 || fail "embed_store $s failed"
    tail -1 "$RES/embed_$s.log"
  fi
done

if [ ! -f "$RES/recall/recall_corpus.summary.json" ]; then
  echo "=== $(date -Iseconds) 3: recall@3 for both stores"
  lim=(); [ "$SMOKE" = 1 ] && lim=(--limit 5)
  for s in corpus wiki; do
    $PY experiments/retrieval_recall.py retrieve "$D/$s" --out "$RES/recall/recall_$s.jsonl" "${lim[@]}" > /dev/null 2>&1 || fail "recall retrieve $s failed"
  done
  VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/retrieval_recall.py grade "$RES/recall/recall_corpus.jsonl" "$RES/recall/recall_wiki.jsonl" \
    > "$RES/recall/grade.log" 2>&1 || fail "recall grading failed"
fi
r3=$(python3 -c "import json;print(json.load(open('$RES/recall/recall_corpus.summary.json'))['recall@3'])")
w3=$(python3 -c "import json;print(json.load(open('$RES/recall/recall_wiki.summary.json'))['recall@3'])")
echo "    recall@3: corpus $r3, wiki $w3 (gate: corpus >= 0.6)"
if [ "$SMOKE" != 1 ] && ! python3 -c "import sys; sys.exit(0 if $r3 >= 0.6 else 1)"; then
  fail "GATE: corpus recall@3 $r3 < 0.6 (pre-registered: stop before training)"
fi

if [ ! -f "$FT/train.jsonl" ]; then
  echo "=== $(date -Iseconds) 4: RAG fine-tune data (corpus store only)"
  $PY scripts/distill/build_rag_ft.py retrieve --store "$D/corpus" --out "$FT" "${RLIM[@]}" > "$RES/rag_retrieve.log" 2>&1 || fail "rag retrieve failed"
  VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I scripts/distill/build_rag_ft.py generate --out "$FT" "${GLIM[@]}" > "$RES/rag_generate.log" 2>&1 \
    || fail "rag generate failed ($RES/rag_generate.log)"
  cat "$FT/report.md"
fi

for seed in 0 1; do
  out=$CK/s$seed
  [ -f "$out/DONE" ] && continue
  mkdir -p "$out"
  echo "=== $(date -Iseconds) 5: RAG fine-tune seed $seed -> $out"
  TRAIN_SCRIPT=finetune_chat.py run_train "$out/train.log" --checkpoint "$BASE10" --split-seed 0 --log-every 50 --conversations 200000 \
    --pilot "$FT" --eval-every 500 --seed "$seed" --seq-len 512 "${FTLIM[@]}" --out "$out"
  [ -f "$out/best_model.pt" ] || fail "RAG fine-tune seed $seed produced no best_model.pt"
  touch "$out/DONE"
done

F=$RES/factuality
mkdir -p "$F"
if [ ! -f "$F/grades.jsonl" ]; then
  echo "=== $(date -Iseconds) 6: factuality generation for every arm"
  gen() { PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u experiments/factuality_gen.py "$@" --out "$F" "${EVLIM[@]}" >> "$F/gen.log" 2>&1; }
  : > "$F/gen.log"
  gen R0="$B10S1" || fail "gen R0"
  gen --append R0b="$B10S0" || fail "gen R0b"
  gen --append --store "$D/corpus" R1c="$B10S1" || fail "gen R1c"
  gen --append --store "$D/corpus" R2c_s0="$CK/s0/best_model.pt" R2c_s1="$CK/s1/best_model.pt" || fail "gen R2c"
  gen --append --store "$D/wiki" R2w_s0="$CK/s0/best_model.pt" R2w_s1="$CK/s1/best_model.pt" || fail "gen R2w"
  gen --append R2closed_s0="$CK/s0/best_model.pt" R2closed_s1="$CK/s1/best_model.pt" || fail "gen R2-closed"
  gen --append --oracle R2oracle_s0="$CK/s0/best_model.pt" R2oracle_s1="$CK/s1/best_model.pt" || fail "gen R2-oracle"
  echo "=== $(date -Iseconds) 6: grading"
  VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/factuality_grade.py "$F" "${GRLIM[@]}" > "$F/grade.log" 2>&1 || fail "grading failed"
fi
$PY experiments/retrieval_report.py "$F" --recall-dir "$RES/recall" || fail "report failed"
[ "$SMOKE" = 1 ] || bash scripts/notify.sh "Retrieval experiment done: see experiments/results/retrieval/factuality/report.md" 3
echo "=== $(date -Iseconds) RETRIEVAL_DONE"
