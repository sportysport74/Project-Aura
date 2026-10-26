#!/bin/sh
# Round 4: sleeps that do work on their own, on top of 'recommended' (seed 0).
# Compare with experiments/round2/recommended.json (2.9609; seed noise ~0.02).
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
mkdir -p experiments/round4/logs
for arm in rec_sleep_prune rec_sleep_shy rec_sleep_replay rec_sleep_replay_rand; do
  $PY experiments/ab_run.py --arm $arm --out round4 > experiments/round4/logs/$arm.log 2>&1 &
done
wait
cp experiments/round2/recommended*.json experiments/round4/
$PY experiments/summarize.py experiments/round4
