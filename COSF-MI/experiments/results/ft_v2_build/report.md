# ft_v2 build report (fact-first)

Source `data/distill_pilot/ft_v1`, teacher Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)). 73.1 min. gsm8k rows without a gold answer: 0.

## Sizes (samples / AURA tokens, 16k-tokenizer count as ft_v1)

| category | train | val | dropped (train/val) |
|---|---|---|---|
| arith | 90,205 / 4.47M | 2,000 / 0.10M | 0 / 0 |
| chat | 48,968 / 4.16M | 1,855 / 0.16M | 3,454 / 145 |
| explain | 110,486 / 10.17M | 1,955 / 0.18M | 2,779 / 45 |
| gsm8k | 4,287 / 0.46M | 256 / 0.03M | 627 / 44 |
| oasst2 | 2,677 / 0.26M | 189 / 0.02M | 113 / 11 |
| story | 52,906 / 8.72M | 2,000 / 0.33M | 0 / 0 |

## Regeneration (per request; a request is dropped when none of its candidates passes)

| category | requests | kept | reasons for dropped requests | rejected candidates (all) |
|---|---|---|---|---|
| chat | 54,422 | 50,823 | simile_first 2679, playful_serious 640, too_many_similes 245, exact_dup 23, latex 7, too_long 3, answer_leak 1, near_dup 1 | simile_first 2615, too_many_similes 1301, playful_serious 475, too_long 34, latex 11, answer_leak 6, exact_dup 6, near_dup 5, truncated 1 |
| explain | 115,265 | 112,441 | simile_first 1872, too_many_similes 586, playful_serious 305, latex 31, exact_dup 9, too_long 8, answer_leak 8, near_dup 2, code 2, truncated 1 | simile_first 3913, too_many_similes 2836, playful_serious 887, too_long 86, latex 43, answer_leak 33, exact_dup 8, near_dup 7, truncated 3, code 2, identity 1 |
| gsm8k | 5,214 | 4,543 | wrong_answer 615, exact_dup 51, answer_not_last 3, near_dup 1, truncated 1 | wrong_answer 866, answer_not_last 18, too_long 5, truncated 2 |
| oasst2 | 2,990 | 2,866 | simile_first 56, code 13, playful_serious 13, latex 10, too_long 10, too_many_similes 10, identity 7, exact_dup 3, near_dup 2 | simile_first 65, too_many_similes 41, playful_serious 31, too_long 21, code 11, identity 10, latex 7, truncated 4, answer_leak 1, exact_dup 1 |

## Style (kept replies, train + val)

| category | comparison anywhere | comparison in 1st sentence | joke marker or '!' | ends in '?' | serious prompts |
|---|---|---|---|---|---|
| arith | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| chat | 16.4% | 0.0% | 57.5% | 25.2% | 3.9% |
| explain | 12.9% | 0.0% | 16.7% | 25.1% | 7.6% |
| gsm8k | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| oasst2 | 6.9% | 0.0% | 10.1% | 27.6% | 13.7% |
| story | 17.0% | 0.3% | 75.7% | 0.0% | 0.0% |
