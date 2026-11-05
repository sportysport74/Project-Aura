# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| v2b_b2_s0 | pieces | greedy | 100 | **0.300** | 13% | 34% | 52% | 1% | 37% | 0 |
| v2b_b2_s0 | pieces | sampled_rep | 300 | **0.252** | 4% | 43% | 53% | 0% | 38% | 0 |
| v2b_b2_s0 | joined | greedy | 100 | **0.280** | 5% | 46% | 49% | 0% | 34% | 0 |
| v2b_b2_s0 | joined | sampled_rep | 300 | **0.230** | 3% | 41% | 57% | 0% | 40% | 0 |
| v2b_b2_s1 | pieces | greedy | 100 | **0.330** | 13% | 40% | 47% | 0% | 31% | 0 |
| v2b_b2_s1 | pieces | sampled_rep | 300 | **0.260** | 5% | 41% | 53% | 0% | 37% | 0 |
| v2b_b2_s1 | joined | greedy | 100 | **0.310** | 10% | 42% | 48% | 0% | 36% | 0 |
| v2b_b2_s1 | joined | sampled_rep | 300 | **0.262** | 5% | 42% | 53% | 0% | 35% | 0 |
| v2b_b5_s0 | pieces | greedy | 100 | **0.350** | 15% | 40% | 45% | 0% | 31% | 0 |
| v2b_b5_s0 | pieces | sampled_rep | 300 | **0.280** | 6% | 44% | 50% | 0% | 33% | 0 |
| v2b_b5_s0 | joined | greedy | 100 | **0.325** | 17% | 31% | 51% | 1% | 30% | 0 |
| v2b_b5_s0 | joined | sampled_rep | 300 | **0.267** | 4% | 45% | 50% | 1% | 35% | 0 |
| v2b_b5_s1 | pieces | greedy | 100 | **0.355** | 14% | 43% | 43% | 0% | 33% | 0 |
| v2b_b5_s1 | pieces | sampled_rep | 300 | **0.288** | 5% | 47% | 47% | 0% | 35% | 0 |
| v2b_b5_s1 | joined | greedy | 100 | **0.365** | 15% | 43% | 41% | 1% | 26% | 0 |
| v2b_b5_s1 | joined | sampled_rep | 300 | **0.253** | 6% | 39% | 54% | 1% | 42% | 0 |
| v2b_b10_s0 | pieces | greedy | 100 | **0.360** | 13% | 46% | 41% | 0% | 26% | 0 |
| v2b_b10_s0 | pieces | sampled_rep | 300 | **0.305** | 7% | 48% | 45% | 0% | 32% | 0 |
| v2b_b10_s0 | joined | greedy | 100 | **0.350** | 13% | 44% | 43% | 0% | 30% | 0 |
| v2b_b10_s0 | joined | sampled_rep | 300 | **0.293** | 8% | 42% | 49% | 1% | 35% | 0 |
| v2b_b10_s1 | pieces | greedy | 100 | **0.400** | 20% | 40% | 40% | 0% | 27% | 0 |
| v2b_b10_s1 | pieces | sampled_rep | 300 | **0.327** | 10% | 45% | 45% | 0% | 32% | 0 |
| v2b_b10_s1 | joined | greedy | 100 | **0.365** | 17% | 39% | 43% | 1% | 25% | 0 |
| v2b_b10_s1 | joined | sampled_rep | 300 | **0.310** | 10% | 42% | 48% | 0% | 33% | 0 |
| control_reference | - | - | 100 | **0.995** | 99% | 1% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 90% | 10% | 32% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| v2b_b2_s0 | pieces | 0.35 | 0.41 | 0.15 | 0.28 | 0.22 | 0.17 | 0.20 | 0.10 |
| v2b_b2_s0 | joined | 0.30 | 0.43 | 0.09 | 0.22 | 0.23 | 0.17 | 0.15 | 0.07 |
| v2b_b2_s1 | pieces | 0.39 | 0.40 | 0.15 | 0.21 | 0.26 | 0.17 | 0.24 | 0.12 |
| v2b_b2_s1 | joined | 0.33 | 0.43 | 0.15 | 0.29 | 0.25 | 0.17 | 0.22 | 0.10 |
| v2b_b5_s0 | pieces | 0.39 | 0.49 | 0.20 | 0.28 | 0.24 | 0.13 | 0.20 | 0.17 |
| v2b_b5_s0 | joined | 0.39 | 0.43 | 0.09 | 0.27 | 0.25 | 0.18 | 0.17 | 0.19 |
| v2b_b5_s1 | pieces | 0.43 | 0.48 | 0.15 | 0.22 | 0.25 | 0.18 | 0.24 | 0.26 |
| v2b_b5_s1 | joined | 0.42 | 0.44 | 0.06 | 0.23 | 0.22 | 0.12 | 0.13 | 0.31 |
| v2b_b10_s0 | pieces | 0.39 | 0.52 | 0.19 | 0.26 | 0.27 | 0.15 | 0.26 | 0.31 |
| v2b_b10_s0 | joined | 0.50 | 0.50 | 0.13 | 0.19 | 0.29 | 0.13 | 0.20 | 0.26 |
| v2b_b10_s1 | pieces | 0.49 | 0.52 | 0.24 | 0.28 | 0.30 | 0.19 | 0.26 | 0.17 |
| v2b_b10_s1 | joined | 0.46 | 0.57 | 0.17 | 0.22 | 0.24 | 0.23 | 0.24 | 0.19 |
