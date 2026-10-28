#!/usr/bin/env bash
# Distill fine-tune A/B: run every args file listed in <queue> one after another, then evaluate.
#   scripts/ft_ab_queue.sh configs/ft_ab/queue.txt [extra finetune args, e.g. --max-steps 50 for a smoke test]
# Each run logs to <its --out>/train.log. Use via launch_train.sh with TRAIN_CMD="bash scripts/ft_ab_queue.sh".
set -euo pipefail
QUEUE="$1"; shift
DIR="$(dirname "$QUEUE")"
CKPTS=()
while read -r f; do
  [ -n "$f" ] || continue
  ARGS="$(cat "$DIR/$f")"
  OUT="$(sed -E 's/.*--out ([^ ]+).*/\1/' <<< "$ARGS")"
  mkdir -p "$OUT"
  echo "=== $(date -Iseconds) $f -> $OUT"
  # shellcheck disable=SC2086
  .venv/bin/python -u finetune_chat.py $ARGS "$@" 2>&1 | tee "$OUT/train.log"
  CKPTS+=("$(basename "$f" .args)=$OUT/best_model.pt")
done < "$QUEUE"
if [ "$#" -eq 0 ]; then  # full runs only, not smoke tests
  echo "=== $(date -Iseconds) eval"
  .venv/bin/python -u experiments/ab_eval.py --out experiments/results/ab_distill "${CKPTS[@]}"
fi
echo "=== $(date -Iseconds) queue done"
