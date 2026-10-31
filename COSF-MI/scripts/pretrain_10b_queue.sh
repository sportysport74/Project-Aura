#!/usr/bin/env bash
# 10B-token pretrain of l100 (experiments/plans/pretrain_10b.md), continued from the 2.2B checkpoint. Two stages, each started by
# hand after the user's go-ahead:
#   STAGE=to5b   main run (WSD, peak 3e-4, warmup 2000) up to step S5, then the 5B branch (linear decay to 0 over 0.5B tokens, so it
#                ends at 5.0B total), then evals of the 2.2B base and the 5B branch (v1 val, factuality, arithmetic battery) and the
#                5B gate (factuality gain >= +0.05 with a paired-bootstrap CI above 0). Then it stops: the user decides on stage 2.
#   STAGE=to10b  main run resumed from latest.pt to the end of the epoch (10B total, decay over the last 10%), then the 10B evals.
# Crash safety: train_overnight.py saves the full state (weights, optimizer, scheduler, data position, RNG) every 60 minutes and at
# every stop, atomically, keeping the previous one as latest.prev.pt; tested by scripts/test_resume.sh (SIGKILL + resume is
# bit-identical in eager mode). Nothing restarts automatically. A marker file (QUEUE_RUNNING) is left behind by a PC crash or a
# kill; the next launch then records the crash in CRASHES.md and refuses to start until relaunched with ACK_CRASH=1 (the user's call).
# Pre-flight: the GPU power cap must read 450 W, else it stops (ask the user; the cap resets on reboot).
# Pause for gaming: scripts/pause_train.sh (state saved, queue exits cleanly); continue: scripts/resume_train.sh.
# Stops on the first failure (set -e) and never retries; a CUDA error is labelled. Run via launch_train.sh with
#   TRAIN_CMD="env STAGE=to5b bash scripts/pretrain_10b_queue.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VPY=~/venvs/vllm/bin/python
STAGE=${STAGE:?set STAGE=to5b or STAGE=to10b}
BASE=checkpoints/pretrain/l100_on_s1/best_model.pt
MAIN=checkpoints/pretrain/l100_10b
BRANCH=checkpoints/pretrain/l100_5b_branch
RES=experiments/results/pretrain_10b
TOK_STEP=16384
S5=$(( (5000000000 - 2200000000 - 500000000) / TOK_STEP ))   # 140,380: the branch starts here...
B5=$(( 500000000 / TOK_STEP ))                                 # ...and decays over 30,517 steps, ending at 5.0B total
mkdir -p "$MAIN" "$BRANCH" "$RES" run_control

# --- pre-flight ---------------------------------------------------------------------------------------------------------------
cap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
if [ "${cap%.*}" != 450 ]; then
  echo "=== $(date -Iseconds) STOPPED: GPU power limit is ${cap} W, not 450 W. Ask the user to re-apply the cap (it resets on reboot)."
  exit 3
fi
MARK=$MAIN/QUEUE_RUNNING
if [ -f "$MARK" ]; then
  {
    echo "## Unclean end detected $(date -Iseconds)"
    echo "- marker: $(cat "$MARK")"
    echo "- last log line: $(tail -n 1 "$MAIN/train.log" 2>/dev/null | cut -c1-200)"
    echo "- latest.pt: $(stat -c '%y' "$MAIN/latest.pt" 2>/dev/null || echo none)"
    echo
  } >> "$MAIN/CRASHES.md"
  if [ "${ACK_CRASH:-0}" != 1 ]; then
    echo "=== $(date -Iseconds) STOPPED: the previous queue run did not end cleanly (PC crash or kill). Recorded in $MAIN/CRASHES.md."
    echo "    Not resuming on my own. After the user's OK, relaunch with ACK_CRASH=1; training resumes from latest.pt."
    exit 2
  fi
  echo "=== crash acknowledged (ACK_CRASH=1), resuming"
fi
echo "stage=$STAGE started=$(date -Iseconds) pid=$$" > "$MARK"
trap 'rm -f "$MARK"' EXIT   # runs on a normal exit or a failure; a BSOD / power loss / SIGKILL leaves the marker

source scripts/lib_train.sh
echo "env STAGE=$STAGE bash scripts/pretrain_10b_queue.sh" > run_control/LAST_QUEUE   # what scripts/resume_train.sh relaunches
train() {  # $1 = log file, rest = args (pause-aware: exit 75 ends the queue cleanly, see scripts/lib_train.sh)
  local log=$1; shift
  run_train "$log" --model-size l100 --corpus data/corpus_v2/tok24k_ext --val-corpus data/corpus_v1/tok24k_v2 \
    --seq-len 512 --batch-size 32 --lr 3e-4 --warmup 2000 --schedule wsd --decay-frac 0.1 --epochs 1 --seed 1 \
    --heartbeat live --no-sleep --log-every 200 --save-every 100000000 --save-every-minutes 60 "$@"
}

evaluate() {  # $1 = point name, rest = NAME=CKPT pairs
  local point=$1; shift
  local out=$RES/factuality_$point
  echo "=== $(date -Iseconds) evals at $point: $*"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u experiments/factuality_gen.py "$@" --out "$out" > "$out.gen.log" 2>&1
  VLLM_USE_FLASHINFER_SAMPLER=0 $VPY -I experiments/factuality_grade.py "$out" > "$out.grade.log" 2>&1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/base_arith_eval.py "$@" --max-new 160 \
    --out "$RES/arith_$point" > "$RES/arith_$point.log" 2>&1
  grep -E "^\| (base|b5|b10|m10) \| joined" "$out/summary.md" || true
}

if [ "$STAGE" = to5b ]; then
  if [ ! -f "$MAIN/stable_step$S5.pt" ]; then
    echo "=== $(date -Iseconds) main run to step $S5 ($(( S5 * TOK_STEP / 1000000 ))M new tokens) -> $MAIN"
    train "$MAIN/train.log" --init-from "$BASE" --resume auto --max-steps "$S5" --checkpoint-dir "$MAIN"
    cp "$MAIN/latest.pt" "$MAIN/stable_step$S5.pt"
  fi
  if [ ! -f "$BRANCH/final_model.pt" ]; then
    echo "=== $(date -Iseconds) 5B branch: steps $S5 -> $(( S5 + B5 )), linear decay to 0 -> $BRANCH"
    rs=(--resume-state "$MAIN/stable_step$S5.pt"); [ -f "$BRANCH/latest.pt" ] && rs=(--resume auto)
    train "$BRANCH/train.log" "${rs[@]}" --decay-start-step "$S5" --decay-steps $(( S5 + B5 )) --max-steps $(( S5 + B5 )) \
      --save-final --eval-batches 100000 --checkpoint-dir "$BRANCH"
    grep -E "val_loss=" "$BRANCH/train.log" | tail -1
  fi
  evaluate 5b base="$BASE" b5="$BRANCH/final_model.pt"
  $PY experiments/factuality_compare.py "$RES/factuality_5b" base b5 | tee "$RES/gate_5b.json"
  echo "=== $(date -Iseconds) STAGE to5b DONE. Gate result above; stage to10b waits for the user."
else
  echo "=== $(date -Iseconds) main run resumed to the end of the epoch (10B total) -> $MAIN"
  train "$MAIN/train.log" --init-from "$BASE" --resume auto --checkpoint-dir "$MAIN"
  grep -E "Val loss" "$MAIN/train.log" | tail -1
  evaluate 10b base="$BASE" b5="$BRANCH/final_model.pt" b10="$MAIN/best_model.pt"
  $PY experiments/factuality_compare.py "$RES/factuality_10b" b5 b10 | tee "$RES/compare_10b_vs_5b.json"
  echo "=== $(date -Iseconds) STAGE to10b DONE"
fi
