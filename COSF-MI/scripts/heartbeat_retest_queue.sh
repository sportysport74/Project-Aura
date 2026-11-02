#!/usr/bin/env bash
# Heartbeat retest on the fixed embedding (experiments/plans/heartbeat_retest.md): aura_std_off at s25 and m50 × seeds 1-3.
# The "on" arm is the aura_std sweep's runs (already done by then). Pause-aware and watchdogged through sweep_queue.sh.
cd "$(dirname "$0")/.." || exit 1
SIZES="s25 m50" ARMS=aura_std_off bash scripts/sweep_queue.sh
