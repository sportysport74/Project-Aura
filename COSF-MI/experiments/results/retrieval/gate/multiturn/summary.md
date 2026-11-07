# Factuality eval (100 explain questions)

Grader Qwen/Qwen2.5-7B-Instruct (Apache-2.0 (Copyright 2024 Alibaba Cloud)), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, mismatched references ~0.

| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| M0_s0 | joined | sampled_rep | 114 | **0.092** | 4% | 10% | 83% | 3% | 38% | 0 |
| M1_s0 | joined | sampled_rep | 114 | **0.355** | 11% | 50% | 39% | 0% | 14% | 0 |
| M0_s1 | joined | sampled_rep | 114 | **0.088** | 2% | 14% | 83% | 1% | 33% | 0 |
| M1_s1 | joined | sampled_rep | 114 | **0.351** | 11% | 49% | 39% | 1% | 12% | 0 |
| control_reference | - | - | 38 | **1.000** | 100% | 0% | 0% | 0% | 0% | 0 |
| control_mismatched | - | - | 38 | **0.000** | 0% | 0% | 87% | 13% | 32% | 0 |

## Per topic (score, sampled_rep, pooled seeds)

| model | encoding | allergies | ancient Egypt | astronauts and space travel | camping | cars | castles | cats | clocks and hours | computers | elephants | glaciers | hot air balloons | keeping cool in summer | levers and pulleys | making a sandwich | mountains | nests and burrows | oceans | orchestras | place value | planting a garden | punctuation | recycling | riddles | robots | staying healthy | submarines | telling odd and even numbers apart | the heart | the internet | the ocean floor | the sun | visiting a museum | wind power | zoos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0_s0 | joined | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.08 | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| M1_s0 | joined | 0.33 | 0.33 | 0.50 | 0.67 | 0.17 | 0.42 | 0.17 | 0.67 | 0.17 | 0.17 | 0.00 | 0.17 | 0.00 | 0.33 | 0.33 | 0.50 | 0.67 | 0.67 | 0.67 | 0.33 | 0.00 | 0.00 | 0.67 | 0.33 | 0.17 | 0.67 | 0.00 | 0.33 | 0.50 | 0.67 | 0.17 | 0.00 | 0.50 | 0.50 | 0.17 |
| M0_s1 | joined | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.17 | 0.50 | 0.00 | 0.00 | 0.08 | 0.67 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 | 0.00 | 0.50 |
| M1_s1 | joined | 1.00 | 0.17 | 0.50 | 0.50 | 0.33 | 0.58 | 0.17 | 0.33 | 0.00 | 0.17 | 0.33 | 0.33 | 0.33 | 0.50 | 0.33 | 0.67 | 0.67 | 0.50 | 0.67 | 0.17 | 0.00 | 0.17 | 0.33 | 0.67 | 0.00 | 0.33 | 0.00 | 0.17 | 0.67 | 0.17 | 0.00 | 0.33 | 0.25 | 0.33 | 0.33 |
