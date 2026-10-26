#!/bin/sh
# Scale contest: which model size / batch gives the best val loss for the same
# wall-clock budget on this PC? Step 1 measures each candidate's real speed ALONE
# on the GPU (300 steps); step 2 gives each the steps it can do in $MINUTES minutes,
# then runs them all in parallel (results depend on steps, not on timing).
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
MINUTES=${MINUTES:-25}
mkdir -p experiments/scale/logs experiments/scale/calib
A="--batch-size 8 --lr 3e-4"
B="--batch-size 32 --lr 6e-4 --warmup 200"
C="--batch-size 32 --lr 5e-4 --warmup 200 --d-model 512 --layers 6 --heads 8 --d-ff 2048"
D="--batch-size 32 --lr 4e-4 --warmup 200 --d-model 512 --layers 8 --heads 8 --d-ff 2048"
for c in A B C D; do
  eval args=\$$c
  $PY experiments/ab_run.py --arm recommended $args --steps 300 --eval-every 100000 --out scale/calib --tag _calib_$c > experiments/scale/calib/$c.log 2>&1
  min=$($PY -c "import json;print(json.load(open('experiments/scale/calib/recommended_calib_$c.json'))['final']['elapsed_min'])")
  steps=$($PY -c "print(int(300 / $min * $MINUTES))")
  echo "$c: 300 steps in $min min -> $steps steps for $MINUTES min"
  eval "steps_$c=$steps"
done
for c in A B C D; do
  eval args=\$$c; eval steps=\$steps_$c
  $PY experiments/ab_run.py --arm recommended $args --steps $steps --eval-every $((steps / 10)) \
    --out scale --tag _$c --save > experiments/scale/logs/$c.log 2>&1 &
done
wait
$PY experiments/summarize.py experiments/scale
