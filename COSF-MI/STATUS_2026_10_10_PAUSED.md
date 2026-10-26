# PAUSED Oct 10, 10:50 AM — user needed the GPU for a game

The smart run (`checkpoints/smart_big_4ep`) was paused on purpose.
- Done: epoch 1 of 4 (crash-recovery state saved 10:46, step 43,008). Epoch 2 was ~2% in; those ~2 min are lost.
- Data/tokenizer: data/mix + models/tokenizer_mix16k (see STATUS_2026_10_08_EVENING.md, "Oct 10" sections).

## To resume (picks up from latest.pt, then runs the chat fine-tune automatically)

    cd C:\Users\fatba\COSF-MI\COSF-MI
    bash run_smart.sh

(run_smart.sh uses `--resume auto`, restarts itself after crashes, and appends to logs/smart_big_4ep.log.
 About 4.8 h of pretraining left, then ~10 min chat fine-tune -> checkpoints/aura_chat_smart.)
Watch with fish.bat.
