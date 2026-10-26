#!/bin/sh
# Forgetting test: stories (6k steps) then Shakespeare (4k steps), 3 arms x 2 seeds.
cd "$(dirname "$0")/.."
PY=../venv/Scripts/python.exe
mkdir -p experiments/forgetting/logs
for seed in 0 1; do
  for arm in nosleep replay_old replay_recent; do
    $PY experiments/forgetting.py --arm $arm --seed $seed > experiments/forgetting/logs/${arm}_seed$seed.log 2>&1 &
  done
  wait
done
$PY - <<'PY'
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob("experiments/forgetting/*.json"))]
print(f"{'run':<22} {'story after A':>13} {'story final':>11} {'forgetting':>10} {'shakespeare':>11} {'fresh B':>8}")
for r in sorted(rows, key=lambda r: (r['arm'], r['name'])):
    print(f"{r['name']:<22} {r['story_after_A']:>13.4f} {r['story_final']:>11.4f} {r['forgetting']:>+10.4f} "
          f"{r['shakespeare_final']:>11.4f} {r['fresh_shakespeare_batches']:>8}")
PY
