# Arithmetic battery on base models (no fine-tune)

Greedy exact match, 410 problems. Prompt `You: {q}\nAura:` as in the corpus. 1-digit cells = fact recall.


## l100_pretrain (`checkpoints/pretrain/l100_on_s1/best_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 100% | 100% |
| + 2-digit | 80% | 80% |
| + 3-digit | 50% | 52% |
| - 1-digit | 100% | 100% |
| - 2-digit | 65% | 75% |
| - 3-digit | 45% | 50% |
| × 1-digit | 100% | 100% |
| × 2-digit | 5% | 5% |
| × 3-digit | 2% | 2% |
| ÷ 1-digit divisor | 35% | 35% |
| ÷ 2-digit divisor | 32% | 32% |
| **ALL** | **53%** | **54%** |

## l100_sweep_on_s1 (`checkpoints/sweep/l100_on_s1/final_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 13% | 13% |
| + 2-digit | 0% | 0% |
| + 3-digit | 0% | 0% |
| - 1-digit | 20% | 20% |
| - 2-digit | 0% | 0% |
| - 3-digit | 0% | 0% |
| × 1-digit | 3% | 7% |
| × 2-digit | 0% | 0% |
| × 3-digit | 0% | 0% |
| ÷ 1-digit divisor | 5% | 2% |
| ÷ 2-digit divisor | 5% | 2% |
| **ALL** | **4%** | **3%** |
