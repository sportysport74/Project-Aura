#!/usr/bin/env bash
# Spike-guard smoke tests (plan aura_v3_base.md, "Step 3 changes"; Jon's list 2026-11-09), end to end through train_overnight.py and
# lib_train.sh's run_train (the real exit-code handling), on s25 / arithmetic with small guard settings so each test takes ~1-2 min:
#   window 50, baseline 200, armed at 300, N = 100, m = 0.5, max 900 steps.
#   T1 obs spike that recovers within N   -> trigger + recovery, no rollback, run finishes
#   T2 obs spike that doesn't recover     -> one rollback (pre-onset checkpoint), data skip to the trigger step, alert, run finishes
#   T3 a second non-recovering spike      -> exit 77, STOP (FAILED marker written by the queue code)
#   T4 a real NaN step (before arming)    -> immediate rollback from the non-finite loss, run finishes
#   T5 a real weight spike (noise 5x each matrix's std after step 491) just before a save that falls between two 10-step checks: the
#      save at 495 holds spiked weights, saves inside the episode are held, and the rollback loads the step-450 save (pre-spike): its
#      weight fingerprint matches the 450 save, not the 495 one.
# ntfy is off (GUARD_NOTIFY=0, mark_failed stubbed); alerts show as "ALERT" lines. Scratch dirs under checkpoints/smoke/guard/.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_train.sh
export GUARD_NOTIFY=0
mark_failed() { echo "$*" > "$D/FAILED_STUB"; }
PY=.venv/bin/python
ROOT=checkpoints/smoke/guard
rm -rf "$ROOT"; mkdir -p "$ROOT"
BASE=(--model-size s25 --projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt --corpus data/arith_settle/tok24k
      --seq-len 512 --batch-size 32 --lr 6e-4 --warmup 100 --epochs 1 --max-steps 900 --decay-steps 900 --eval-batches 5 --seed 1
      --no-sleep --log-every 10 --save-final --resume auto
      --spike-guard --guard-margin 0.5 --guard-window 50 --guard-baseline 200 --guard-arm 300 --guard-recover 100)
t() {  # name save_every inject
  D=$ROOT/$1; mkdir -p "$D"
  echo "=== $(date +%T) $1 (save every $2, inject $3)"
  ( run_train "$D/train.log" "${BASE[@]}" --checkpoint-dir "$D" --save-every "$2" --guard-inject "$3" ); echo "rc=$?" > "$D/rc"
  cat "$D/rc"
}
t T1_recover 100 "obs:500:60:1.0"
t T2_rollback 100 "obs:500:400:1.0"
t T3_second 100 "obs:500:400:1.0,obs:700:400:1.0:always"
t T4_nan 100 "nan:250"
t T5_savewindow 45 "weights:491:5.0"
$PY -m pytest -q tests/test_spike_guard_smoke.py 2>&1 | tail -15
