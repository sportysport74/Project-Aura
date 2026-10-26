#!/bin/sh
# Round 5: context chirality with the gate starting at 0 (free sign), 3 seeds + bridge-word probe.
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
mkdir -p experiments/round5/logs
for seed in 0 1 2; do
  $PY experiments/ab_run.py --arm context_chir0 --seed $seed --out round5 --save \
    > experiments/round5/logs/context_chir0_seed$seed.log 2>&1 &
done
wait
cp experiments/round2/recommended*.json experiments/round3/context_chir*.json experiments/round5/
$PY experiments/summarize.py experiments/round5
for s in "" _seed1 _seed2; do $PY experiments/bridge_words.py experiments/round5/context_chir0$s.pt; done
