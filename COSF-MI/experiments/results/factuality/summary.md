# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| base | pieces | greedy | 100 | **0.030** | 1% | 4% | 72% | 23% | 20% | 0 |
| base | pieces | sampled_rep | 300 | **0.058** | 1% | 10% | 77% | 12% | 37% | 0 |
| base | joined | greedy | 100 | **0.030** | 0% | 6% | 87% | 7% | 19% | 0 |
| base | joined | sampled_rep | 300 | **0.063** | 0% | 13% | 84% | 4% | 36% | 0 |
| ft_s0 | pieces | greedy | 100 | **0.105** | 2% | 17% | 77% | 4% | 33% | 0 |
| ft_s0 | pieces | sampled_rep | 300 | **0.140** | 1% | 27% | 73% | 0% | 35% | 0 |
| ft_s0 | joined | greedy | 100 | **0.130** | 2% | 22% | 73% | 3% | 33% | 0 |
| ft_s0 | joined | sampled_rep | 300 | **0.147** | 1% | 27% | 69% | 2% | 36% | 0 |
| ft_s1 | pieces | greedy | 100 | **0.100** | 2% | 16% | 76% | 6% | 30% | 0 |
| ft_s1 | pieces | sampled_rep | 300 | **0.127** | 1% | 23% | 75% | 1% | 39% | 0 |
| ft_s1 | joined | greedy | 100 | **0.100** | 2% | 16% | 79% | 3% | 31% | 0 |
| ft_s1 | joined | sampled_rep | 300 | **0.142** | 1% | 27% | 72% | 1% | 34% | 0 |
| control_reference | - | - | 100 | **1.000** | 100% | 0% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 90% | 10% | 31% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| base | pieces | 0.08 | 0.06 | 0.09 | 0.03 | 0.08 | 0.03 | 0.07 | 0.00 |
| base | joined | 0.06 | 0.16 | 0.00 | 0.00 | 0.12 | 0.04 | 0.06 | 0.00 |
| ft_s0 | pieces | 0.18 | 0.22 | 0.00 | 0.16 | 0.13 | 0.06 | 0.15 | 0.19 |
| ft_s0 | joined | 0.19 | 0.21 | 0.06 | 0.17 | 0.13 | 0.10 | 0.09 | 0.19 |
| ft_s1 | pieces | 0.19 | 0.24 | 0.02 | 0.08 | 0.17 | 0.03 | 0.11 | 0.10 |
| ft_s1 | joined | 0.23 | 0.24 | 0.06 | 0.14 | 0.15 | 0.09 | 0.04 | 0.07 |
