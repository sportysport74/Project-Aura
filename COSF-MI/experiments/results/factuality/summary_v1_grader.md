# First-pass grader results (superseded; inflated by PARTIAL, see spot_check_v1_grader.md)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| base | pieces | greedy | 100 | **0.130** | 0% | 26% | 69% | 5% | 57% | 0 |
| base | pieces | sampled_rep | 300 | **0.120** | 0% | 24% | 73% | 3% | 73% | 0 |
| base | joined | greedy | 100 | **0.180** | 0% | 36% | 64% | 0% | 63% | 0 |
| base | joined | sampled_rep | 300 | **0.197** | 0% | 39% | 60% | 1% | 61% | 0 |
| ft_s0 | pieces | greedy | 100 | **0.270** | 2% | 50% | 45% | 3% | 46% | 0 |
| ft_s0 | pieces | sampled_rep | 300 | **0.328** | 0% | 66% | 34% | 0% | 39% | 0 |
| ft_s0 | joined | greedy | 100 | **0.295** | 1% | 57% | 40% | 2% | 43% | 0 |
| ft_s0 | joined | sampled_rep | 300 | **0.313** | 0% | 63% | 37% | 0% | 40% | 0 |
| ft_s1 | pieces | greedy | 100 | **0.250** | 1% | 48% | 48% | 3% | 48% | 0 |
| ft_s1 | pieces | sampled_rep | 300 | **0.337** | 0% | 67% | 32% | 1% | 33% | 0 |
| ft_s1 | joined | greedy | 100 | **0.245** | 0% | 49% | 48% | 3% | 49% | 0 |
| ft_s1 | joined | sampled_rep | 300 | **0.322** | 0% | 64% | 34% | 1% | 39% | 0 |
| control_reference | - | - | 100 | **0.995** | 99% | 1% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 95% | 5% | 89% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| base | pieces | 0.13 | 0.24 | 0.13 | 0.11 | 0.14 | 0.01 | 0.06 | 0.07 |
| base | joined | 0.15 | 0.27 | 0.15 | 0.11 | 0.31 | 0.18 | 0.17 | 0.14 |
| ft_s0 | pieces | 0.38 | 0.45 | 0.28 | 0.28 | 0.35 | 0.23 | 0.28 | 0.31 |
| ft_s0 | joined | 0.37 | 0.41 | 0.20 | 0.32 | 0.36 | 0.22 | 0.24 | 0.26 |
| ft_s1 | pieces | 0.40 | 0.45 | 0.24 | 0.38 | 0.37 | 0.22 | 0.22 | 0.26 |
| ft_s1 | joined | 0.36 | 0.40 | 0.30 | 0.36 | 0.40 | 0.23 | 0.20 | 0.17 |
