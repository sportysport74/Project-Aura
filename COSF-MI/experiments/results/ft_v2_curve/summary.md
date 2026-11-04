# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| ft2_b2_s0 | pieces | greedy | 100 | **0.190** | 8% | 22% | 69% | 1% | 42% | 0 |
| ft2_b2_s0 | pieces | sampled_rep | 300 | **0.157** | 1% | 29% | 70% | 0% | 49% | 0 |
| ft2_b2_s0 | joined | greedy | 100 | **0.140** | 3% | 22% | 72% | 3% | 38% | 0 |
| ft2_b2_s0 | joined | sampled_rep | 300 | **0.147** | 1% | 27% | 71% | 0% | 44% | 0 |
| ft2_b2_s1 | pieces | greedy | 100 | **0.180** | 7% | 22% | 70% | 0% | 36% | 1 |
| ft2_b2_s1 | pieces | sampled_rep | 300 | **0.175** | 1% | 34% | 66% | 0% | 45% | 0 |
| ft2_b2_s1 | joined | greedy | 100 | **0.110** | 2% | 18% | 78% | 2% | 35% | 0 |
| ft2_b2_s1 | joined | sampled_rep | 300 | **0.165** | 2% | 28% | 68% | 1% | 43% | 0 |
| ft2_b5_s0 | pieces | greedy | 100 | **0.240** | 7% | 34% | 59% | 0% | 34% | 0 |
| ft2_b5_s0 | pieces | sampled_rep | 300 | **0.168** | 1% | 31% | 67% | 1% | 49% | 0 |
| ft2_b5_s0 | joined | greedy | 100 | **0.225** | 8% | 29% | 62% | 1% | 33% | 0 |
| ft2_b5_s0 | joined | sampled_rep | 300 | **0.177** | 2% | 32% | 65% | 1% | 44% | 0 |
| ft2_b5_s1 | pieces | greedy | 100 | **0.235** | 6% | 35% | 59% | 0% | 38% | 0 |
| ft2_b5_s1 | pieces | sampled_rep | 300 | **0.197** | 2% | 36% | 62% | 1% | 46% | 0 |
| ft2_b5_s1 | joined | greedy | 100 | **0.210** | 8% | 26% | 65% | 1% | 35% | 0 |
| ft2_b5_s1 | joined | sampled_rep | 300 | **0.183** | 3% | 31% | 66% | 0% | 46% | 0 |
| ft2_b10_s0 | pieces | greedy | 100 | **0.270** | 12% | 30% | 58% | 0% | 41% | 0 |
| ft2_b10_s0 | pieces | sampled_rep | 300 | **0.230** | 2% | 43% | 55% | 1% | 39% | 0 |
| ft2_b10_s0 | joined | greedy | 100 | **0.250** | 13% | 24% | 63% | 0% | 39% | 0 |
| ft2_b10_s0 | joined | sampled_rep | 300 | **0.213** | 2% | 39% | 59% | 1% | 40% | 0 |
| ft2_b10_s1 | pieces | greedy | 100 | **0.255** | 10% | 31% | 59% | 0% | 47% | 0 |
| ft2_b10_s1 | pieces | sampled_rep | 300 | **0.240** | 4% | 40% | 56% | 0% | 41% | 0 |
| ft2_b10_s1 | joined | greedy | 100 | **0.280** | 9% | 38% | 51% | 2% | 35% | 0 |
| ft2_b10_s1 | joined | sampled_rep | 300 | **0.198** | 2% | 35% | 63% | 0% | 42% | 0 |
| control_reference | - | - | 100 | **1.000** | 100% | 0% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 90% | 10% | 32% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| ft2_b2_s0 | pieces | 0.26 | 0.31 | 0.04 | 0.13 | 0.15 | 0.08 | 0.07 | 0.07 |
| ft2_b2_s0 | joined | 0.23 | 0.27 | 0.11 | 0.13 | 0.09 | 0.08 | 0.09 | 0.12 |
| ft2_b2_s1 | pieces | 0.26 | 0.29 | 0.06 | 0.12 | 0.22 | 0.13 | 0.13 | 0.05 |
| ft2_b2_s1 | joined | 0.21 | 0.27 | 0.06 | 0.23 | 0.15 | 0.09 | 0.07 | 0.12 |
| ft2_b5_s0 | pieces | 0.24 | 0.32 | 0.04 | 0.16 | 0.19 | 0.10 | 0.11 | 0.02 |
| ft2_b5_s0 | joined | 0.25 | 0.29 | 0.13 | 0.12 | 0.18 | 0.15 | 0.13 | 0.05 |
| ft2_b5_s1 | pieces | 0.26 | 0.34 | 0.13 | 0.17 | 0.20 | 0.17 | 0.13 | 0.02 |
| ft2_b5_s1 | joined | 0.23 | 0.34 | 0.09 | 0.17 | 0.17 | 0.14 | 0.17 | 0.02 |
| ft2_b10_s0 | pieces | 0.37 | 0.36 | 0.17 | 0.22 | 0.25 | 0.08 | 0.15 | 0.10 |
| ft2_b10_s0 | joined | 0.30 | 0.39 | 0.17 | 0.14 | 0.24 | 0.15 | 0.15 | 0.00 |
| ft2_b10_s1 | pieces | 0.30 | 0.43 | 0.19 | 0.21 | 0.23 | 0.17 | 0.17 | 0.10 |
| ft2_b10_s1 | joined | 0.24 | 0.40 | 0.17 | 0.13 | 0.18 | 0.13 | 0.17 | 0.07 |
