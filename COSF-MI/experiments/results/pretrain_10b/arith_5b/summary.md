# Arithmetic battery on base models (no fine-tune)

Greedy exact match, 410 problems, reply cap 160 tokens. Prompt `You: {q}\nAura:` as in the corpus. 1-digit cells = fact recall.


## base (`checkpoints/pretrain/l100_on_s1/best_model.pt`)

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

## b5 (`checkpoints/pretrain/l100_5b_branch/final_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 100% | 100% |
| + 2-digit | 98% | 98% |
| + 3-digit | 95% | 92% |
| - 1-digit | 100% | 100% |
| - 2-digit | 80% | 88% |
| - 3-digit | 78% | 78% |
| × 1-digit | 100% | 100% |
| × 2-digit | 8% | 12% |
| × 3-digit | 2% | 2% |
| ÷ 1-digit divisor | 50% | 52% |
| ÷ 2-digit divisor | 72% | 70% |
| **ALL** | **69%** | **70%** |
