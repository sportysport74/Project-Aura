#!/bin/bash
# Wait for the teacher data run (scripts/generate_cosf_distill.py) to finish, free the GPU,
# chat fine-tune smart_big_10ep on TinyChat + the verified data, then run the math exam.
# fish.bat follows along: the fine-tune log becomes the newest log when it starts.
cd "$(dirname "$0")"
PY=../venv/Scripts/python.exe
DATA_LOG=logs/distill_10k.log
FT_LOG=logs/aura_chat_distill.log
export PYTHONIOENCODING=utf-8

restart_generator() {
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object CommandLine -match 'generate_cosf_distill' | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    sleep 5
    $PY -u scripts/generate_cosf_distill.py --n 10000 > $DATA_LOG 2> logs/distill_10k.err.log &
}

# 1. Wait for the data. If the log goes quiet for 15 min the generator is stuck or dead:
#    restart it (it resumes from the samples already saved).
while ! grep -q "^Done\." $DATA_LOG; do
    age=$(( $(date +%s) - $(stat -c %Y $DATA_LOG) ))
    if [ $age -gt 900 ]; then
        echo "$(date) data log quiet for ${age}s, restarting the generator" >> logs/distill_10k_history.log
        restart_generator
    fi
    sleep 60
done
grep "^Done\." $DATA_LOG >> logs/distill_10k_history.log

# 2. Unload the teacher and wait for its VRAM to clear.
ollama stop qwen2.5:32b
for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 6000 ] && break
    sleep 5
done

# 3. Fine-tune: same recipe as aura_chat_smart10, plus 30% verified teacher data.
$PY finetune_chat.py --checkpoint checkpoints/smart_big_10ep/best_model.pt \
    --distill data/cosf_distill_verified.jsonl --out checkpoints/aura_chat_distill > $FT_LOG 2>&1

# 4. Math exam, old vs new (written outside logs/ so fish keeps showing the fine-tune).
$PY experiments/math_eval.py checkpoints/aura_chat_smart10/best_model.pt \
    checkpoints/aura_chat_distill/best_model.pt > experiments/results/math_eval_summary.txt 2>&1
