#!/usr/bin/env bash
# Rerun of the ft_l100 eval after the base arm crashed in math_eval (pretrain checkpoint has no chat_format).
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u experiments/ab_eval.py --out experiments/results/ft_l100 \
  --vocab 24000 --seq-len 512 --base checkpoints/pretrain/l100_on_s1/best_model.pt base=checkpoints/pretrain/l100_on_s1/best_model.pt l100_s0=checkpoints/ft_l100_s0/best_model.pt l100_s1=checkpoints/ft_l100_s1/best_model.pt
echo "=== $(date -Iseconds) queue done"
