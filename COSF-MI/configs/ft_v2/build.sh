#!/usr/bin/env bash
# Builds data/distill_pilot/ft_v2 (fact-first rewrite of ft_v1). Launch args (the launcher's config path) are ignored.
set -euo pipefail
VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/build_factfirst.py --out data/distill_pilot/ft_v2
echo "=== $(date -Iseconds) queue done"
