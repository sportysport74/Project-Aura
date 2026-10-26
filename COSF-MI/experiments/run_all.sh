#!/bin/sh
# Run every A/B arm, 3 at a time. Logs in experiments/results/logs/.
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
mkdir -p experiments/results/logs
run() { $PY experiments/ab_run.py "$@" > "experiments/results/logs/$1$2$3.log" 2>&1; }
(run --arm baseline; run --arm live_01; run --arm sleep_consol; run --arm chir_nowd) &
(run --arm baseline --seed 1; run --arm live_01_t03; run --arm sleep_decay0) &
(run --arm live_001; run --arm sleep_gentle; run --arm sleep_phi) &
wait
echo ALL DONE
