# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| base | pieces | greedy | 100 | **0.030** | 1% | 4% | 72% | 23% | 21% | 0 |
| base | pieces | sampled_rep | 300 | **0.057** | 1% | 10% | 78% | 11% | 37% | 0 |
| base | joined | greedy | 100 | **0.025** | 0% | 5% | 88% | 7% | 21% | 0 |
| base | joined | sampled_rep | 300 | **0.062** | 0% | 12% | 83% | 4% | 36% | 0 |
| b5 | pieces | greedy | 100 | **0.065** | 1% | 11% | 86% | 2% | 44% | 0 |
| b5 | pieces | sampled_rep | 300 | **0.060** | 0% | 11% | 75% | 14% | 49% | 0 |
| b5 | joined | greedy | 100 | **0.020** | 0% | 4% | 94% | 2% | 8% | 0 |
| b5 | joined | sampled_rep | 300 | **0.017** | 0% | 3% | 95% | 2% | 16% | 0 |
| control_reference | - | - | 100 | **0.995** | 99% | 1% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 100 | **0.000** | 0% | 0% | 89% | 11% | 31% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | animals | biology | chemistry | earth | physics | space | tech | world |
|---|---|---|---|---|---|---|---|---|---|
| base | pieces | 0.07 | 0.06 | 0.09 | 0.03 | 0.08 | 0.03 | 0.07 | 0.00 |
| base | joined | 0.06 | 0.16 | 0.00 | 0.00 | 0.12 | 0.04 | 0.04 | 0.00 |
| b5 | pieces | 0.06 | 0.15 | 0.00 | 0.04 | 0.02 | 0.04 | 0.07 | 0.10 |
| b5 | joined | 0.04 | 0.04 | 0.00 | 0.00 | 0.00 | 0.03 | 0.02 | 0.00 |
