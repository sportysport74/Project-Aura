#!/bin/sh
# Round 3: per-token (context-dependent) chirality vs the round-2 'recommended' arm.
# Seeds 0-2 of 'recommended' are already in experiments/round2 (runs are deterministic).
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
mkdir -p experiments/round3/logs
for seed in 0 1 2; do
  $PY experiments/ab_run.py --arm context_chir --seed $seed --out round3 --save \
    > experiments/round3/logs/context_chir_seed$seed.log 2>&1 &
done
wait
cp experiments/round2/recommended*.json experiments/round2/baseline*.json experiments/round3/
$PY experiments/summarize.py experiments/round3
