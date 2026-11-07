# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | pieces | greedy | 100 | **0.390** | 18% | 42% | 40% | 0% | 30% | 0 |
| R0 | pieces | sampled_rep | 300 | **0.323** | 10% | 45% | 45% | 0% | 33% | 0 |
| R0 | joined | greedy | 100 | **0.370** | 18% | 38% | 43% | 1% | 27% | 0 |
| R0 | joined | sampled_rep | 300 | **0.308** | 10% | 42% | 48% | 0% | 33% | 0 |
| R0b | pieces | greedy | 100 | **0.365** | 14% | 45% | 41% | 0% | 26% | 0 |
| R0b | pieces | sampled_rep | 300 | **0.302** | 6% | 48% | 45% | 0% | 32% | 0 |
| R0b | joined | greedy | 100 | **0.350** | 14% | 42% | 44% | 0% | 31% | 0 |
| R0b | joined | sampled_rep | 300 | **0.295** | 8% | 43% | 48% | 1% | 34% | 0 |
| R1c | pieces | greedy | 100 | **0.240** | 10% | 28% | 48% | 14% | 11% | 0 |
| R1c | pieces | sampled_rep | 300 | **0.242** | 6% | 36% | 46% | 12% | 18% | 0 |
| R1c | joined | greedy | 100 | **0.120** | 4% | 16% | 29% | 51% | 7% | 0 |
| R1c | joined | sampled_rep | 300 | **0.172** | 3% | 28% | 41% | 28% | 11% | 0 |
| R2c_s0 | pieces | greedy | 100 | **0.585** | 30% | 57% | 13% | 0% | 9% | 0 |
| R2c_s0 | pieces | sampled_rep | 300 | **0.510** | 21% | 61% | 19% | 0% | 11% | 0 |
| R2c_s0 | joined | greedy | 100 | **0.475** | 16% | 63% | 21% | 0% | 13% | 0 |
| R2c_s0 | joined | sampled_rep | 300 | **0.520** | 24% | 56% | 20% | 0% | 12% | 0 |
| R2c_s1 | pieces | greedy | 100 | **0.530** | 27% | 52% | 21% | 0% | 16% | 0 |
| R2c_s1 | pieces | sampled_rep | 300 | **0.515** | 19% | 65% | 16% | 0% | 8% | 0 |
| R2c_s1 | joined | greedy | 100 | **0.530** | 27% | 52% | 21% | 0% | 18% | 0 |
| R2c_s1 | joined | sampled_rep | 300 | **0.505** | 23% | 56% | 22% | 0% | 15% | 0 |
| R2w_s0 | pieces | greedy | 100 | **0.505** | 28% | 45% | 27% | 0% | 19% | 0 |
| R2w_s0 | pieces | sampled_rep | 300 | **0.468** | 16% | 61% | 23% | 0% | 17% | 0 |
| R2w_s0 | joined | greedy | 100 | **0.505** | 27% | 47% | 26% | 0% | 18% | 0 |
| R2w_s0 | joined | sampled_rep | 300 | **0.438** | 15% | 58% | 27% | 0% | 20% | 0 |
| R2w_s1 | pieces | greedy | 100 | **0.545** | 27% | 55% | 18% | 0% | 18% | 0 |
| R2w_s1 | pieces | sampled_rep | 300 | **0.458** | 12% | 68% | 20% | 0% | 16% | 0 |
| R2w_s1 | joined | greedy | 100 | **0.475** | 24% | 47% | 29% | 0% | 14% | 0 |
| R2w_s1 | joined | sampled_rep | 300 | **0.442** | 14% | 61% | 25% | 0% | 20% | 0 |
| R2closed_s0 | pieces | greedy | 100 | **0.360** | 13% | 46% | 41% | 0% | 25% | 0 |
| R2closed_s0 | pieces | sampled_rep | 300 | **0.297** | 3% | 53% | 44% | 0% | 36% | 0 |
| R2closed_s0 | joined | greedy | 100 | **0.355** | 14% | 43% | 42% | 1% | 28% | 0 |
| R2closed_s0 | joined | sampled_rep | 300 | **0.298** | 6% | 47% | 46% | 1% | 36% | 0 |
| R2closed_s1 | pieces | greedy | 100 | **0.385** | 14% | 49% | 37% | 0% | 26% | 0 |
| R2closed_s1 | pieces | sampled_rep | 300 | **0.278** | 4% | 47% | 49% | 0% | 34% | 0 |
| R2closed_s1 | joined | greedy | 100 | **0.355** | 11% | 49% | 40% | 0% | 24% | 0 |
| R2closed_s1 | joined | sampled_rep | 300 | **0.307** | 4% | 53% | 43% | 0% | 29% | 0 |
| R2oracle_s0 | pieces | greedy | 100 | **0.645** | 44% | 41% | 15% | 0% | 10% | 0 |
| R2oracle_s0 | pieces | sampled_rep | 300 | **0.505** | 22% | 56% | 21% | 0% | 17% | 0 |
| R2oracle_s0 | joined | greedy | 100 | **0.615** | 40% | 43% | 17% | 0% | 11% | 0 |
| R2oracle_s0 | joined | sampled_rep | 300 | **0.505** | 24% | 54% | 22% | 0% | 17% | 0 |
| R2oracle_s1 | pieces | greedy | 100 | **0.635** | 42% | 43% | 15% | 0% | 10% | 0 |
| R2oracle_s1 | pieces | sampled_rep | 300 | **0.493** | 19% | 61% | 20% | 0% | 15% | 0 |
| R2oracle_s1 | joined | greedy | 100 | **0.605** | 41% | 39% | 20% | 0% | 11% | 0 |
| R2oracle_s1 | joined | sampled_rep | 300 | **0.487** | 22% | 54% | 24% | 0% | 18% | 0 |
| control_reference | - | - | 100 | **0.995** | 99% | 1% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 88% | 12% | 31% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| R0 | pieces | 0.49 | 0.51 | 0.24 | 0.27 | 0.30 | 0.19 | 0.26 | 0.17 |
| R0 | joined | 0.46 | 0.56 | 0.15 | 0.23 | 0.24 | 0.22 | 0.26 | 0.19 |
| R0b | pieces | 0.39 | 0.51 | 0.19 | 0.26 | 0.27 | 0.15 | 0.26 | 0.29 |
| R0b | joined | 0.50 | 0.50 | 0.13 | 0.19 | 0.30 | 0.14 | 0.20 | 0.24 |
| R1c | pieces | 0.15 | 0.20 | 0.33 | 0.30 | 0.29 | 0.26 | 0.19 | 0.19 |
| R1c | joined | 0.13 | 0.18 | 0.19 | 0.26 | 0.15 | 0.17 | 0.15 | 0.14 |
| R2c_s0 | pieces | 0.50 | 0.56 | 0.43 | 0.49 | 0.55 | 0.55 | 0.37 | 0.57 |
| R2c_s0 | joined | 0.50 | 0.56 | 0.48 | 0.53 | 0.58 | 0.50 | 0.39 | 0.55 |
| R2c_s1 | pieces | 0.54 | 0.58 | 0.52 | 0.47 | 0.53 | 0.50 | 0.39 | 0.57 |
| R2c_s1 | joined | 0.51 | 0.54 | 0.46 | 0.48 | 0.52 | 0.56 | 0.35 | 0.57 |
| R2w_s0 | pieces | 0.54 | 0.62 | 0.43 | 0.46 | 0.44 | 0.41 | 0.43 | 0.29 |
| R2w_s0 | joined | 0.50 | 0.55 | 0.35 | 0.39 | 0.44 | 0.40 | 0.43 | 0.36 |
| R2w_s1 | pieces | 0.55 | 0.56 | 0.41 | 0.44 | 0.41 | 0.41 | 0.44 | 0.36 |
| R2w_s1 | joined | 0.48 | 0.58 | 0.37 | 0.42 | 0.39 | 0.40 | 0.48 | 0.33 |
| R2closed_s0 | pieces | 0.40 | 0.52 | 0.13 | 0.28 | 0.26 | 0.22 | 0.26 | 0.10 |
| R2closed_s0 | joined | 0.36 | 0.47 | 0.26 | 0.30 | 0.31 | 0.15 | 0.26 | 0.12 |
| R2closed_s1 | pieces | 0.35 | 0.53 | 0.17 | 0.23 | 0.24 | 0.19 | 0.24 | 0.12 |
| R2closed_s1 | joined | 0.43 | 0.47 | 0.26 | 0.28 | 0.27 | 0.22 | 0.24 | 0.14 |
| R2oracle_s0 | pieces | 0.57 | 0.66 | 0.50 | 0.50 | 0.49 | 0.44 | 0.35 | 0.40 |
| R2oracle_s0 | joined | 0.54 | 0.60 | 0.46 | 0.57 | 0.48 | 0.41 | 0.35 | 0.57 |
| R2oracle_s1 | pieces | 0.56 | 0.57 | 0.39 | 0.48 | 0.51 | 0.49 | 0.44 | 0.38 |
| R2oracle_s1 | joined | 0.57 | 0.59 | 0.44 | 0.50 | 0.43 | 0.40 | 0.37 | 0.55 |
