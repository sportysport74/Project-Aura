#!/usr/bin/env bash
# Resume test for train_overnight.py (2026-10-31): a run killed with SIGKILL mid-run and resumed with --resume auto must end
# bit-identical to the same run done straight through. Eager (--no-compile), because compiled runs aren't bit-reproducible.
# Covers weights, optimizer, scheduler (WSD), data position and RNG (dropout), plus the time-based save.
# Then a compiled check at the real batch (32): kill + resume must continue the loss curve (not bitwise).
# Usage (from COSF-MI/): bash scripts/test_resume.sh [size] [steps]   (default l100, 120)
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=${1:-l100}; STEPS=${2:-120}; KILL_AT=$((STEPS * 6 / 100 * 10))   # a logged step (log every 10)
OUT=${RESUME_TEST_DIR:-checkpoints/resume_test}
rm -rf "$OUT"; mkdir -p "$OUT"
common=(--model-size "$SIZE" --corpus data/corpus_v1/tok24k_v2 --seq-len 512 --batch-size 32 --lr 3e-4 --warmup 20
        --schedule wsd --decay-steps "$STEPS" --max-steps "$STEPS" --epochs 1 --seed 1 --heartbeat live --no-sleep
        --log-every 10 --save-every 25 --save-every-minutes 0.25 --save-final)
[ -f checkpoints/pretrain/l100_on_s1/best_model.pt ] && [ "$SIZE" = l100 ] && common+=(--init-from checkpoints/pretrain/l100_on_s1/best_model.pt)

run_pair() {  # $1 = label, rest = extra args
  local label=$1; shift
  echo "== $label: straight"
  .venv/bin/python -u train_overnight.py "${common[@]}" "$@" --checkpoint-dir "$OUT/${label}_straight" > "$OUT/${label}_straight.log" 2>&1
  echo "== $label: killed at step >= $KILL_AT, then resumed"
  .venv/bin/python -u train_overnight.py "${common[@]}" "$@" --checkpoint-dir "$OUT/${label}_crash" > "$OUT/${label}_crash1.log" 2>&1 &
  local pid=$!
  until grep -aqE "(^|[^0-9])step=$KILL_AT " "$OUT/${label}_crash1.log" 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "run exited before the kill point"; exit 1; }; sleep 0.5; done
  kill -9 $pid; wait $pid 2>/dev/null || true
  echo "   killed; last save: $(grep -o 'saved latest.pt at step [0-9]*' "$OUT/${label}_crash1.log" | tail -1)"
  .venv/bin/python -u train_overnight.py "${common[@]}" "$@" --checkpoint-dir "$OUT/${label}_crash" --resume auto > "$OUT/${label}_crash2.log" 2>&1
  grep -E "Resuming full|RNG|Resuming epoch" "$OUT/${label}_crash2.log" | sed 's/^/   /'
}

run_pair eager --no-compile --deterministic --precision fp32 --batch-size 8   # fp32: bf16 eager uses the flash kernel, whose backward is nondeterministic (~1e-4 loss noise); batch 8: eager l100 at 32 needs 34.7 GiB
.venv/bin/python - "$OUT" "$STEPS" <<'PY'
import re, sys, torch; sys.path.insert(0, "src/python")
out, steps = sys.argv[1], int(sys.argv[2])
a = torch.load(f"{out}/eager_straight/final_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
b = torch.load(f"{out}/eager_crash/final_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
diff = [k for k in a if not torch.equal(a[k], b[k])]
loss = lambda f: dict(re.findall(r"step=(\d+) loss=(\S+)", open(f).read()))
la, lb = loss(f"{out}/eager_straight.log"), {**loss(f"{out}/eager_crash1.log"), **loss(f"{out}/eager_crash2.log")}
mism = [s for s in la if la[s] != lb.get(s)]
sa = torch.load(f"{out}/eager_straight/latest.pt", map_location="cpu", weights_only=False)
sb = torch.load(f"{out}/eager_crash/latest.pt", map_location="cpu", weights_only=False)
opt_eq = all(torch.equal(x, y) for x, y in zip(
    [t for s in sa["optimizer_state_dict"]["state"].values() for t in s.values() if torch.is_tensor(t)],
    [t for s in sb["optimizer_state_dict"]["state"].values() for t in s.values() if torch.is_tensor(t)]))
print(f"EAGER: weights {'IDENTICAL' if not diff else f'DIFFER in {len(diff)} tensors, e.g. {diff[:3]}'}; "
      f"optimizer state {'IDENTICAL' if opt_eq else 'DIFFERS'}; scheduler step {sa['scheduler_state_dict']['last_epoch']} vs "
      f"{sb['scheduler_state_dict']['last_epoch']}; logged losses {len(la) - len(mism)}/{len(la)} equal"
      + (f" (first mismatch at step {mism[0]})" if mism else ""))
sys.exit(1 if diff or mism or not opt_eq else 0)
PY
run_pair compiled
.venv/bin/python - "$OUT" <<'PY'
import re, sys
out = sys.argv[1]
loss = lambda f: {int(s): float(v) for s, v in re.findall(r"step=(\d+) loss=(\S+)", open(f).read())}
a = loss(f"{out}/compiled_straight.log"); b = {**loss(f"{out}/compiled_crash1.log"), **loss(f"{out}/compiled_crash2.log")}
gaps = {s: round(b[s] - a[s], 4) for s in sorted(a) if s in b}
print("COMPILED: loss gap (resumed - straight) per logged step:", gaps)
print("COMPILED: max |gap|", max(abs(g) for g in gaps.values()))
PY
echo "RESUME_TEST_DONE"
