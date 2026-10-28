# Decoding check: is the looping the decoding or the model?

Same ~400 prompts as the A/B report; sampled rows pool 2 samples per prompt. Greedy is from the A/B run.


## replies with a repeated 4-gram

| checkpoint | greedy | topk40 | topp | topp_rep1.15 |
|---|---|---|---|---|
| new_s0 | 57% | 31% | 37% | 6% |
| new_s1 | 55% | 34% | 39% | 7% |
| old_s0 | 5% | 3% | 3% | 1% |
| old_s1 | 5% | 2% | 3% | 1% |
| ref_aura_chat_distill | 4% | 2% | 3% | 1% |

## replies with a 4-gram 3+ times

| checkpoint | greedy | topk40 | topp | topp_rep1.15 |
|---|---|---|---|---|
| new_s0 | 22% | 7% | 12% | 1% |
| new_s1 | 24% | 9% | 12% | 1% |
| old_s0 | 2% | 1% | 1% | 0% |
| old_s1 | 3% | 1% | 1% | 0% |
| ref_aura_chat_distill | 1% | 1% | 1% | 0% |

## distinct-2 per reply

| checkpoint | greedy | topk40 | topp | topp_rep1.15 |
|---|---|---|---|---|
| new_s0 | 0.811 | 0.906 | 0.882 | 0.979 |
| new_s1 | 0.800 | 0.903 | 0.881 | 0.978 |
| old_s0 | 0.965 | 0.983 | 0.979 | 0.996 |
| old_s1 | 0.961 | 0.983 | 0.979 | 0.995 |
| ref_aura_chat_distill | 0.972 | 0.984 | 0.981 | 0.997 |

## mean words

| checkpoint | greedy | topk40 | topp | topp_rep1.15 |
|---|---|---|---|---|
| new_s0 | 47 | 52 | 53 | 59 |
| new_s1 | 48 | 54 | 52 | 59 |
| old_s0 | 14 | 14 | 14 | 15 |
| old_s1 | 14 | 14 | 14 | 15 |
| ref_aura_chat_distill | 13 | 14 | 14 | 15 |

## turn ended

| checkpoint | greedy | topk40 | topp | topp_rep1.15 |
|---|---|---|---|---|
| new_s0 | 96% | 98% | 96% | 98% |
| new_s1 | 95% | 97% | 97% | 98% |
| old_s0 | 98% | 99% | 99% | 99% |
| old_s1 | 98% | 99% | 99% | 99% |
| ref_aura_chat_distill | 99% | 99% | 99% | 99% |

## ends in "?" (chat/explain)

| checkpoint | greedy | topk40 | topp | topp_rep1.15 |
|---|---|---|---|---|
| new_s0 | 6% | 18% | 19% | 64% |
| new_s1 | 4% | 21% | 17% | 62% |
| old_s0 | 6% | 8% | 9% | 11% |
| old_s1 | 8% | 8% | 8% | 10% |
| ref_aura_chat_distill | 8% | 7% | 8% | 13% |
