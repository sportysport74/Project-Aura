# Arithmetic battery on base models (no fine-tune)

Greedy exact match, 410 problems, reply cap 160 tokens. Prompt `You: {q}\nAura:` as in the corpus. 1-digit cells = fact recall.


## b2 (`checkpoints/v2base/A/best_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 100% | 100% |
| + 2-digit | 92% | 92% |
| + 3-digit | 90% | 90% |
| - 1-digit | 100% | 100% |
| - 2-digit | 85% | 88% |
| - 3-digit | 62% | 70% |
| × 1-digit | 100% | 100% |
| × 2-digit | 12% | 18% |
| × 3-digit | 2% | 5% |
| ÷ 1-digit divisor | 55% | 52% |
| ÷ 2-digit divisor | 62% | 62% |
| **ALL** | **67%** | **69%** |

## b5 (`checkpoints/v2base/branch5b/final_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 100% | 100% |
| + 2-digit | 98% | 98% |
| + 3-digit | 92% | 95% |
| - 1-digit | 100% | 100% |
| - 2-digit | 92% | 92% |
| - 3-digit | 78% | 75% |
| × 1-digit | 100% | 100% |
| × 2-digit | 20% | 12% |
| × 3-digit | 2% | 2% |
| ÷ 1-digit divisor | 68% | 68% |
| ÷ 2-digit divisor | 82% | 82% |
| **ALL** | **74%** | **73%** |

## b10 (`checkpoints/v2base/main/best_model.pt`)

| cell | what_is | equals |
|---|---|---|
| + 1-digit | 100% | 100% |
| + 2-digit | 95% | 98% |
| + 3-digit | 95% | 92% |
| - 1-digit | 100% | 100% |
| - 2-digit | 92% | 92% |
| - 3-digit | 80% | 80% |
| × 1-digit | 100% | 100% |
| × 2-digit | 20% | 18% |
| × 3-digit | 2% | 2% |
| ÷ 1-digit divisor | 82% | 82% |
| ÷ 2-digit divisor | 82% | 82% |
| **ALL** | **76%** | **75%** |
