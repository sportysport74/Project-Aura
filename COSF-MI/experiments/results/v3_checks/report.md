# Aura v3 base: scaling check and lr check (rules pre-registered in `experiments/plans/aura_v3_base.md`)

200M tokens per run (3,052 steps × 65,536 tokens), seq 1024, vanilla2, val loss on the whole v1 val split at 1024 tokens (not comparable with the 512-token numbers of earlier bases).

## Step 1: scaling check (l100 lr 6e-4, x330 lr 3e-4)

| seed | l100 | x330 | x330 − l100 |
|---|---|---|---|
| 1 | 3.2967 | 3.2497 | -0.0470 |
| 2 | 3.3025 | 3.2452 | -0.0573 |

**Rule (x330 lower on both seeds): PASS.**

## Step 2: lr check at x330 (seed 1)

| lr | val loss |
|---|---|
| 2e-4 | 3.3907 |
| 3e-4 | 3.2497 |
| 4.5e-4 | 3.1570 |
| 6e-4 | 3.1301 **(best)** |
| 9e-4 | 3.7107 |

Grid edge after the 4-point grid: high. **Best lr so far: 6e-4.**

## Throughput and memory (median tok/s after warm-up; torch peak)

| run | tok/s | peak GiB |
|---|---|---|
| lr_x330_2e-4 | 64,140 | 22.12 |
| lr_x330_4.5e-4 | 62,363 | 22.12 |
| lr_x330_6e-4 | 59,584 | 22.12 |
| lr_x330_9e-4 | 59,348 | 22.12 |
| scale_l100_s1 | 181,976 | 9.47 |
| scale_l100_s2 | 179,249 | 9.47 |
| scale_x330_s1 | 62,172 | 22.12 |
| scale_x330_s2 | 64,017 | 22.12 |
