# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | pieces | greedy | 100 | **0.395** | 19% | 41% | 40% | 0% | 29% | 0 |
| R0 | pieces | sampled_rep | 300 | **0.327** | 10% | 45% | 45% | 0% | 33% | 0 |
| R0 | joined | greedy | 100 | **0.375** | 18% | 39% | 42% | 1% | 26% | 0 |
| R0 | joined | sampled_rep | 300 | **0.310** | 10% | 42% | 48% | 0% | 32% | 0 |
| R0b | pieces | greedy | 100 | **0.360** | 14% | 44% | 42% | 0% | 27% | 0 |
| R0b | pieces | sampled_rep | 300 | **0.307** | 6% | 49% | 44% | 0% | 31% | 0 |
| R0b | joined | greedy | 100 | **0.345** | 13% | 43% | 44% | 0% | 30% | 0 |
| R0b | joined | sampled_rep | 300 | **0.292** | 7% | 44% | 48% | 1% | 35% | 0 |
| R1c | pieces | greedy | 100 | **0.245** | 10% | 29% | 47% | 14% | 11% | 0 |
| R1c | pieces | sampled_rep | 300 | **0.112** | 2% | 18% | 63% | 16% | 22% | 0 |
| R1c | joined | greedy | 100 | **0.125** | 5% | 15% | 30% | 50% | 7% | 0 |
| R1c | joined | sampled_rep | 300 | **0.070** | 1% | 11% | 51% | 36% | 16% | 0 |
| R2c_s0 | pieces | greedy | 100 | **0.595** | 31% | 57% | 12% | 0% | 8% | 0 |
| R2c_s0 | pieces | sampled_rep | 300 | **0.395** | 12% | 56% | 33% | 0% | 27% | 0 |
| R2c_s0 | joined | greedy | 100 | **0.470** | 16% | 62% | 22% | 0% | 14% | 0 |
| R2c_s0 | joined | sampled_rep | 300 | **0.368** | 8% | 57% | 35% | 0% | 25% | 0 |
| R2c_s1 | pieces | greedy | 100 | **0.535** | 28% | 51% | 21% | 0% | 15% | 0 |
| R2c_s1 | pieces | sampled_rep | 300 | **0.393** | 11% | 57% | 32% | 0% | 25% | 0 |
| R2c_s1 | joined | greedy | 100 | **0.525** | 27% | 51% | 22% | 0% | 19% | 0 |
| R2c_s1 | joined | sampled_rep | 300 | **0.383** | 10% | 56% | 33% | 0% | 26% | 0 |
| R2w_s0 | pieces | greedy | 100 | **0.500** | 27% | 46% | 27% | 0% | 18% | 0 |
| R2w_s0 | pieces | sampled_rep | 300 | **0.392** | 11% | 57% | 32% | 0% | 26% | 0 |
| R2w_s0 | joined | greedy | 100 | **0.510** | 27% | 48% | 25% | 0% | 18% | 0 |
| R2w_s0 | joined | sampled_rep | 300 | **0.328** | 10% | 46% | 44% | 0% | 33% | 0 |
| R2w_s1 | pieces | greedy | 100 | **0.550** | 27% | 56% | 17% | 0% | 18% | 0 |
| R2w_s1 | pieces | sampled_rep | 300 | **0.393** | 12% | 55% | 33% | 0% | 30% | 0 |
| R2w_s1 | joined | greedy | 100 | **0.475** | 24% | 47% | 29% | 0% | 14% | 0 |
| R2w_s1 | joined | sampled_rep | 300 | **0.335** | 10% | 47% | 43% | 0% | 34% | 0 |
| R2closed_s0 | pieces | greedy | 100 | **0.365** | 13% | 47% | 40% | 0% | 24% | 0 |
| R2closed_s0 | pieces | sampled_rep | 300 | **0.297** | 3% | 53% | 44% | 0% | 36% | 0 |
| R2closed_s0 | joined | greedy | 100 | **0.360** | 15% | 42% | 42% | 1% | 27% | 0 |
| R2closed_s0 | joined | sampled_rep | 300 | **0.297** | 6% | 48% | 46% | 1% | 35% | 0 |
| R2closed_s1 | pieces | greedy | 100 | **0.380** | 14% | 48% | 38% | 0% | 27% | 0 |
| R2closed_s1 | pieces | sampled_rep | 300 | **0.285** | 5% | 47% | 48% | 0% | 34% | 0 |
| R2closed_s1 | joined | greedy | 100 | **0.350** | 11% | 48% | 41% | 0% | 24% | 0 |
| R2closed_s1 | joined | sampled_rep | 300 | **0.308** | 4% | 53% | 42% | 0% | 29% | 0 |
| R2oracle_s0 | pieces | greedy | 100 | **0.635** | 43% | 41% | 16% | 0% | 11% | 0 |
| R2oracle_s0 | pieces | sampled_rep | 300 | **0.337** | 8% | 51% | 41% | 0% | 32% | 0 |
| R2oracle_s0 | joined | greedy | 100 | **0.610** | 39% | 44% | 17% | 0% | 11% | 0 |
| R2oracle_s0 | joined | sampled_rep | 300 | **0.250** | 5% | 40% | 54% | 1% | 35% | 0 |
| R2oracle_s1 | pieces | greedy | 100 | **0.635** | 42% | 43% | 15% | 0% | 10% | 0 |
| R2oracle_s1 | pieces | sampled_rep | 300 | **0.310** | 6% | 51% | 43% | 0% | 32% | 0 |
| R2oracle_s1 | joined | greedy | 100 | **0.615** | 42% | 39% | 19% | 0% | 11% | 0 |
| R2oracle_s1 | joined | sampled_rep | 300 | **0.273** | 6% | 43% | 51% | 0% | 38% | 0 |
| control_reference | - | - | 100 | **0.995** | 99% | 1% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 88% | 12% | 32% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| R0 | pieces | 0.49 | 0.51 | 0.24 | 0.27 | 0.31 | 0.19 | 0.26 | 0.19 |
| R0 | joined | 0.46 | 0.57 | 0.15 | 0.23 | 0.24 | 0.22 | 0.26 | 0.19 |
| R0b | pieces | 0.40 | 0.52 | 0.19 | 0.26 | 0.27 | 0.15 | 0.26 | 0.31 |
| R0b | joined | 0.50 | 0.49 | 0.13 | 0.19 | 0.30 | 0.14 | 0.20 | 0.21 |
| R1c | pieces | 0.08 | 0.14 | 0.15 | 0.09 | 0.13 | 0.17 | 0.09 | 0.00 |
| R1c | joined | 0.07 | 0.11 | 0.07 | 0.06 | 0.03 | 0.12 | 0.06 | 0.02 |
| R2c_s0 | pieces | 0.45 | 0.49 | 0.44 | 0.34 | 0.40 | 0.37 | 0.28 | 0.29 |
| R2c_s0 | joined | 0.43 | 0.48 | 0.44 | 0.29 | 0.38 | 0.33 | 0.24 | 0.26 |
| R2c_s1 | pieces | 0.46 | 0.51 | 0.44 | 0.31 | 0.39 | 0.37 | 0.30 | 0.26 |
| R2c_s1 | joined | 0.54 | 0.50 | 0.31 | 0.31 | 0.38 | 0.32 | 0.24 | 0.36 |
| R2w_s0 | pieces | 0.49 | 0.55 | 0.33 | 0.33 | 0.33 | 0.29 | 0.43 | 0.31 |
| R2w_s0 | joined | 0.40 | 0.46 | 0.33 | 0.30 | 0.25 | 0.26 | 0.33 | 0.24 |
| R2w_s1 | pieces | 0.40 | 0.57 | 0.46 | 0.39 | 0.33 | 0.28 | 0.39 | 0.24 |
| R2w_s1 | joined | 0.44 | 0.42 | 0.33 | 0.31 | 0.32 | 0.26 | 0.31 | 0.19 |
| R2closed_s0 | pieces | 0.40 | 0.52 | 0.13 | 0.28 | 0.26 | 0.22 | 0.26 | 0.10 |
| R2closed_s0 | joined | 0.35 | 0.46 | 0.28 | 0.29 | 0.32 | 0.15 | 0.26 | 0.12 |
| R2closed_s1 | pieces | 0.36 | 0.52 | 0.17 | 0.27 | 0.23 | 0.21 | 0.24 | 0.14 |
| R2closed_s1 | joined | 0.43 | 0.47 | 0.26 | 0.28 | 0.28 | 0.21 | 0.26 | 0.14 |
| R2oracle_s0 | pieces | 0.40 | 0.48 | 0.35 | 0.36 | 0.28 | 0.24 | 0.26 | 0.21 |
| R2oracle_s0 | joined | 0.29 | 0.29 | 0.35 | 0.32 | 0.20 | 0.22 | 0.15 | 0.12 |
| R2oracle_s1 | pieces | 0.32 | 0.43 | 0.24 | 0.32 | 0.27 | 0.28 | 0.26 | 0.29 |
| R2oracle_s1 | joined | 0.27 | 0.44 | 0.28 | 0.24 | 0.19 | 0.23 | 0.24 | 0.29 |
