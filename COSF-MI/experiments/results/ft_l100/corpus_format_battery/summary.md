# Arithmetic battery on base models (no fine-tune)

Greedy exact match, 410 problems, reply cap 160 tokens. Prompt `You: {q}\nAura:` as in the corpus. 1-digit cells = fact recall.


## l100_base (`checkpoints/pretrain/l100_on_s1/best_model.pt`)

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

## ft_l100_s0 (`checkpoints/ft_l100_s0/best_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 90% | 90% |
| + 2-digit | 70% | 48% |
| + 3-digit | 50% | 42% |
| - 1-digit | 97% | 97% |
| - 2-digit | 78% | 75% |
| - 3-digit | 52% | 57% |
| × 1-digit | 83% | 83% |
| × 2-digit | 2% | 2% |
| × 3-digit | 0% | 0% |
| ÷ 1-digit divisor | 25% | 25% |
| ÷ 2-digit divisor | 35% | 38% |
| **ALL** | **50%** | **48%** |

## ft_l100_s1 (`checkpoints/ft_l100_s1/best_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 97% | 93% |
| + 2-digit | 65% | 50% |
| + 3-digit | 60% | 42% |
| - 1-digit | 100% | 100% |
| - 2-digit | 72% | 70% |
| - 3-digit | 55% | 55% |
| × 1-digit | 87% | 90% |
| × 2-digit | 5% | 5% |
| × 3-digit | 0% | 0% |
| ÷ 1-digit divisor | 30% | 25% |
| ÷ 2-digit divisor | 38% | 42% |
| **ALL** | **52%** | **49%** |
