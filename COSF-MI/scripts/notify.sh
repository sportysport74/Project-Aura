#!/usr/bin/env bash
# Phone/desktop alert via ntfy.sh (2026-11-06). The topic is a random name in run_control/ntfy_topic (gitignored; never commit it: anyone who
# knows the name can read the topic). Messages carry no paths or secrets. Usage: bash scripts/notify.sh "message" [priority 1-5]
# Silently does nothing if the topic file is missing or the network is down (an alert must never break a queue).
cd "$(dirname "$0")/.." || exit 0
[ -f run_control/ntfy_topic ] || exit 0
curl -s -m 10 -H "Title: AURA" -H "Priority: ${2:-3}" -d "$1" "https://ntfy.sh/$(cat run_control/ntfy_topic)" > /dev/null 2>&1 || true
