#!/bin/bash
# Pretrain the smart model with crash auto-restart, then chat fine-tune it.
cd "$(dirname "$0")"
PY=../venv/Scripts/python.exe
LOG=logs/smart_big_4ep.log
export PYTHONIOENCODING=utf-8
for attempt in $(seq 1 30); do
    echo "=== attempt $attempt $(date) ===" >> $LOG
    $PY train_overnight.py --model-size big --epochs 4 --batch-size 32 --lr 5e-4 --warmup 200 \
        --data-dir data/mix --tokenizer models/tokenizer_mix16k.model \
        --checkpoint-dir checkpoints/smart_big_4ep --resume auto >> $LOG 2>&1
    tr '\r' '\n' < $LOG | grep -q "TRAINING COMPLETE" && break
    echo "=== crashed, restarting in 30 s ===" >> $LOG
    sleep 30
done
tr '\r' '\n' < $LOG | grep -cE "^=== crashed" | sed 's/^/crashes: /'
tr '\r' '\n' < $LOG | grep -E "Val loss|TRAINING COMPLETE" | tail -5
$PY finetune_chat.py --checkpoint checkpoints/smart_big_4ep/best_model.pt --out checkpoints/aura_chat_smart > logs/aura_chat_smart.log 2>&1
grep -E "^Base|^Before|^Done|Traceback" logs/aura_chat_smart.log
