#!/bin/sh
# Round 2: golden-angle phases now wrapped at init (model fix). Check the fix
# from the model alone, with phases kept out of weight decay, and with the live heartbeat.
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
mkdir -p experiments/round2/logs
for arm in baseline phase_nowd recommended; do
  $PY experiments/ab_run.py --arm $arm --out round2 > experiments/round2/logs/$arm.log 2>&1 &
done
wait
$PY experiments/summarize.py experiments/round2
